"""Fractional-order capacitor degradation UI."""

from __future__ import annotations

import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLineEdit,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    Figure = None  # type: ignore
    FigureCanvas = None  # type: ignore

TRAIN_COLOR = "#6b7c8c"
FORECAST_OBS_COLOR = "#a0765b"
FORECAST_MEAN_COLOR = "#2f4858"
BIAS_MEAN_COLOR = "#496a7a"
EPISTEMIC_COLOR = "#8ea8ba"
TOTAL_COLOR = "#b7c6d0"
CONFORMAL_COLOR = "#d9cbb0"
HYBRID_COLOR = "#b38b44"
BAR_MAIN_COLOR = "#4c6578"
BAR_TOTAL_COLOR = "#9badbc"

MODEL_OPTIONS = [
    ("Fractional kinetics (FK)", "fk"),
    ("Classical exponential", "classical"),
    ("KWW stretched exponential", "kww"),
]

ANALYSIS_OPTIONS = [
    ("Forecast & UQ", "forecast"),
    ("Sensitivity Study", "sensitivity"),
]

from fractional_core import FractionalConfig, FractionalPICPCore
from surrogate_models import fit_classical, fit_kww
from core import load_dataset, PICPCore
from fractional_sensitivity import (
    BetaPrior,
    LogNormalPrior,
    LogUniformPrior,
    qoi_capacitance,
    qoi_deficit,
    qoi_failure_time,
    sobol_analysis,
)

class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class AnalysisWorker(QRunnable):
    def __init__(self, fn: Callable[..., Dict[str, Any]], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # pragma: no cover - worker errors surface via signal
            message = "".join(traceback.format_exception_only(type(exc), exc)).strip() or str(exc)
            self.signals.error.emit(message)
        else:
            self.signals.finished.emit(result)



def _draw_sobol_panel(
    ax: "plt.Axes",
    S: np.ndarray,
    ST: np.ndarray,
    params: list[str],
    S_ci: Optional[np.ndarray] = None,
    ST_ci: Optional[np.ndarray] = None,
    title: str = "",
) -> None:
    """
    Draw grouped Sobol bars with confidence intervals on a single axis.

    This helper consolidates Sobol plotting logic for both forecast-backed
    and prior-sweep modes, ensuring consistent styling and spacing.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axis for drawing.
    S : np.ndarray
        First-order Sobol indices (shape: n_params).
    ST : np.ndarray
        Total Sobol indices (shape: n_params).
    params : list[str]
        Parameter names for x-axis labels.
    S_ci : np.ndarray | None
        95% CI for first-order indices (shape: 2 × n_params), optional.
    ST_ci : np.ndarray | None
        95% CI for total indices (shape: 2 × n_params), optional.
    title : str
        Panel title (e.g., QoI name).
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    x = np.arange(len(params))
    width = 0.36  # Bar width for grouped bars

    # Draw bars with improved spacing
    bars_main = ax.bar(
        x - width / 2,
        S,
        width,
        label="First-order (S)",
        color=BAR_MAIN_COLOR,
        alpha=0.85,
        edgecolor="#2c3e50",
        linewidth=0.5,
    )
    bars_total = ax.bar(
        x + width / 2,
        ST,
        width,
        label="Total (ST)",
        color=BAR_TOTAL_COLOR,
        alpha=0.75,
        edgecolor="#34495e",
        linewidth=0.5,
    )

    # Add error bars with improved visibility
    if S_ci is not None and S_ci.shape[0] == 2:
        lower, upper = S_ci
        yerr = np.vstack([S - lower, upper - S])
        ax.errorbar(
            x - width / 2,
            S,
            yerr=yerr,
            fmt="none",
            ecolor="#2c3e50",
            elinewidth=1.2,
            capsize=5,
            capthick=1.2,
            label="95% CI",
        )

    if ST_ci is not None and ST_ci.shape[0] == 2:
        lower, upper = ST_ci
        yerr = np.vstack([ST - lower, upper - ST])
        ax.errorbar(
            x + width / 2,
            ST,
            yerr=yerr,
            fmt="none",
            ecolor="#34495e",
            elinewidth=1.2,
            capsize=5,
            capthick=1.2,
        )

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(params, fontsize=10)
    ax.set_ylabel("Sobol Index", fontsize=11)
    ax.set_ylim(0.0, 1.05)  # Sobol indices are in [0, 1]
    ax.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.8)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Legend positioned to avoid bar overlap
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        edgecolor="#888888",
    )

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


class FractionalWindow(QMainWindow):
    """Application window for fractional-kinetics forecasting."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fractional-Order Capacitor Prognostics")
        self.resize(1180, 820)

        self.output_dir = Path.cwd()
        self._data: Optional[pd.DataFrame] = None
        self._time_column: Optional[str] = None
        self._target_column: Optional[str] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._time_values: Optional[np.ndarray] = None
        self._value_values: Optional[np.ndarray] = None
        self.sens_data: Optional[Dict[str, Any]] = None
        self.forecast_data: Optional[Dict[str, Any]] = None
        self.current_mode: str = "forecast"
        self.sensitivity_tab: Optional[QWidget] = None
        self._current_model: str = "fk"

        self._active_worker: Optional[AnalysisWorker] = None
        self.progress_bar: Optional[QProgressBar] = None
        self._pending_model: Optional[str] = None
        self._pending_times: Optional[np.ndarray] = None
        self._pending_values: Optional[np.ndarray] = None
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(2)

        self.uq_group = None
        self.sens_group = None
        self._build_ui()
        self._apply_palette()

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        file_row = QHBoxLayout()
        self.file_label = QLabel("No data loaded")
        self.file_label.setWordWrap(True)
        file_row.addWidget(self.file_label, stretch=1)
        load_button = QPushButton("Load data…")
        load_button.clicked.connect(self.open_file)
        file_row.addWidget(load_button)
        root.addLayout(file_row)

        out_row = QHBoxLayout()
        self.output_dir_label = QLabel(f"Output directory: {self.output_dir}")
        self.output_dir_label.setWordWrap(True)
        out_row.addWidget(self.output_dir_label, stretch=1)
        choose_button = QPushButton("Choose output folder…")
        choose_button.clicked.connect(self._choose_output_dir)
        out_row.addWidget(choose_button)
        root.addLayout(out_row)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Time column:"))
        self.time_combo = QComboBox()
        self.time_combo.setEnabled(False)
        selector_row.addWidget(self.time_combo)
        selector_row.addWidget(QLabel("Target column:"))
        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)
        selector_row.addWidget(self.target_combo)
        selector_row.addStretch(1)
        root.addLayout(selector_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Analysis:"))
        self.mode_combo = QComboBox()
        for label, key in ANALYSIS_OPTIONS:
            self.mode_combo.addItem(label, key)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for label, key in MODEL_OPTIONS:
            self.model_combo.addItem(label, key)
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        options_row.addWidget(self.model_combo)
        options_row.addWidget(QLabel("Training fraction:"))
        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0.4, 0.9)
        self.train_spin.setSingleStep(0.05)
        self.train_spin.setValue(0.7)
        self.train_spin.setDecimals(2)
        options_row.addWidget(self.train_spin)

        options_row.addWidget(QLabel("Confidence:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.6, 0.99)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setValue(0.90)
        self.confidence_spin.setDecimals(2)
        options_row.addWidget(self.confidence_spin)

        options_row.addStretch(1)
        root.addLayout(options_row)

        uq_group = QGroupBox("Uncertainty settings")
        uq_layout = QHBoxLayout(uq_group)
        uq_layout.setSpacing(10)
        uq_layout.addWidget(QLabel("Bootstrap draws:"))
        self.bootstrap_spin = QSpinBox()
        self.bootstrap_spin.setRange(0, 5000)
        self.bootstrap_spin.setSingleStep(50)
        self.bootstrap_spin.setValue(512)
        uq_layout.addWidget(self.bootstrap_spin)

        self.mcmc_check = QCheckBox("Enable MCMC sampling")
        self.mcmc_check.stateChanged.connect(self._toggle_mcmc_controls)
        uq_layout.addWidget(self.mcmc_check)

        uq_layout.addWidget(QLabel("Draws:"))
        self.mcmc_draws_spin = QSpinBox()
        self.mcmc_draws_spin.setRange(100, 10000)
        self.mcmc_draws_spin.setSingleStep(100)
        self.mcmc_draws_spin.setValue(1000)
        uq_layout.addWidget(self.mcmc_draws_spin)

        uq_layout.addWidget(QLabel("Burn-in:"))
        self.mcmc_burnin_spin = QSpinBox()
        self.mcmc_burnin_spin.setRange(0, 5000)
        self.mcmc_burnin_spin.setSingleStep(50)
        self.mcmc_burnin_spin.setValue(300)
        uq_layout.addWidget(self.mcmc_burnin_spin)

        uq_layout.addWidget(QLabel("Step scale:"))
        self.mcmc_step_spin = QDoubleSpinBox()
        self.mcmc_step_spin.setRange(0.05, 5.0)
        self.mcmc_step_spin.setSingleStep(0.05)
        self.mcmc_step_spin.setValue(0.50)
        self.mcmc_step_spin.setDecimals(2)
        uq_layout.addWidget(self.mcmc_step_spin)
        sampler_label = QLabel("Sampler: Random-walk Metropolis")
        sampler_label.setToolTip("Current build uses a Metropolis sampler; HMC/NUTS planned once available.")
        uq_layout.addWidget(sampler_label)
        uq_layout.addStretch(1)
        root.addWidget(uq_group)
        self.uq_group = uq_group
        self._toggle_mcmc_controls()

        sens_group = QGroupBox("Sensitivity")
        sens_row = QHBoxLayout(sens_group)
        sens_row.setSpacing(10)
        self.sensitivity_check = QCheckBox("Compute Sobol indices with FK forecast")
        self.sensitivity_check.stateChanged.connect(self._apply_sensitivity_state)
        sens_row.addWidget(self.sensitivity_check)
        self.sensitivity_summary = QLabel("H: n/a  |  q: n/a  |  N: n/a  |  B: n/a")
        self.sensitivity_summary.setToolTip("Sobol configuration shared by forecast runs and the prior-driven study.")
        sens_row.addWidget(self.sensitivity_summary, stretch=1)
        self.sensitivity_config_btn = QPushButton("Edit settings...")
        self.sensitivity_config_btn.clicked.connect(self._open_sensitivity_tab)
        sens_row.addWidget(self.sensitivity_config_btn)
        root.addWidget(sens_group)
        self.sens_group = sens_group
        self._apply_sensitivity_state()

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run analysis")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_analysis)
        run_row.addWidget(self.run_button)
        self.comparison_btn = QPushButton("Generate baseline comparison")
        self.comparison_btn.setEnabled(False)
        self.comparison_btn.clicked.connect(self._generate_baseline_comparison)
        run_row.addWidget(self.comparison_btn)
        run_row.addStretch(1)
        root.addLayout(run_row)

        save_row = QHBoxLayout()
        self.save_plot_btn = QPushButton("Save forecast plot…")
        self.save_plot_btn.setEnabled(False)
        self.save_plot_btn.clicked.connect(self._save_plot)
        save_row.addWidget(self.save_plot_btn)
        if MATPLOTLIB_AVAILABLE:
            self.save_sensitivity_btn = QPushButton("Save sensitivity plot…")
            self.save_sensitivity_btn.setEnabled(False)
            self.save_sensitivity_btn.clicked.connect(self._save_sensitivity_plot)
            save_row.addWidget(self.save_sensitivity_btn)
        self.export_btn = QPushButton("Export forecast…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_forecast)
        save_row.addWidget(self.export_btn)
        save_row.addStretch(1)
        root.addLayout(save_row)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self._build_overview_tab()
        self._build_sensitivity_tab()
        self._build_forecast_tab()

        status_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        status_row.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        root.addLayout(status_row)
        self._model_changed()
        self._mode_changed()

    def _build_overview_tab(self) -> None:
        overview = QWidget()
        layout = QVBoxLayout(overview)
        layout.setSpacing(10)

        self.param_table = QTableWidget(0, 2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Estimate"])
        self.param_table.verticalHeader().setVisible(False)
        self.param_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.param_table)

        self.metric_table = QTableWidget(0, 2)
        self.metric_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metric_table.verticalHeader().setVisible(False)
        self.metric_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.metric_table)

        self.diagnostics_text = QPlainTextEdit()
        self.diagnostics_text.setReadOnly(True)
        layout.addWidget(self.diagnostics_text, stretch=1)

        self.tabs.addTab(overview, "Overview")


    def _build_sensitivity_tab(self) -> None:
        """
        Build a unified Sensitivity tab showing forecast-backed Sobol indices.

        This tab displays Sobol sensitivity analysis results from the FK forecast.
        The priors are automatically anchored at the MAP fit from the data, and
        Sobol indices quantify parameter influence on downstream QoIs like
        capacitance predictions and failure times.
        """
        sens = QWidget()
        self.sensitivity_tab = sens
        layout = QVBoxLayout(sens)
        layout.setSpacing(12)

        # Explanatory header
        info_group = QGroupBox("About Sensitivity Analysis")
        info_layout = QVBoxLayout(info_group)
        info_text = QLabel(
            "Sobol indices quantify parameter influence on forecast quantities of interest (QoIs). "
            "Run the FK forecast with 'Sobol indices' enabled in the Overview tab, then load results below. "
            "Priors are anchored at the MAP fit from your data, ensuring physically-informed sensitivity."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_group)

        # Sobol configuration (shared with forecast)
        config_group = QGroupBox("Sobol configuration")
        config_form = QFormLayout(config_group)
        config_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.sobol_horizons_edit = QLineEdit("200.0")
        self.sobol_horizons_edit.setPlaceholderText("Comma-separated horizons (e.g., 200, 400)")
        self.sobol_horizons_edit.setClearButtonEnabled(True)
        self.sobol_horizons_edit.setToolTip(
            "Time horizons (hours) for capacitance predictions. "
            "Example: '200, 400' evaluates sensitivity at 200h and 400h."
        )
        config_form.addRow("Horizons", self.sobol_horizons_edit)

        self.thresholds_edit = QLineEdit("0.80, 0.70")
        self.thresholds_edit.setPlaceholderText("0 < q < 1 (e.g., 0.85, 0.70)")
        self.thresholds_edit.setClearButtonEnabled(True)
        self.thresholds_edit.setToolTip(
            "Failure thresholds as fractions of initial capacitance. "
            "Example: '0.80, 0.70' computes time-to-80% and time-to-70%."
        )
        config_form.addRow("Failure thresholds", self.thresholds_edit)

        self.sobol_samples_spin = QSpinBox()
        self.sobol_samples_spin.setRange(128, 16384)
        self.sobol_samples_spin.setSingleStep(128)
        self.sobol_samples_spin.setValue(2048)
        self.sobol_samples_spin.setToolTip(
            "Number of Sobol samples for variance-based decomposition. "
            "Higher values improve accuracy but increase computation time. Recommended: 2048-4096."
        )
        config_form.addRow("Sobol samples", self.sobol_samples_spin)

        self.sobol_bootstrap_spin = QSpinBox()
        self.sobol_bootstrap_spin.setRange(10, 2000)
        self.sobol_bootstrap_spin.setSingleStep(10)
        self.sobol_bootstrap_spin.setValue(200)
        self.sobol_bootstrap_spin.setToolTip(
            "Bootstrap iterations for confidence intervals on Sobol indices. "
            "Recommended: 200 for stable CI estimates."
        )
        config_form.addRow("Bootstrap draws", self.sobol_bootstrap_spin)

        layout.addWidget(config_group)

        self.sobol_horizons_edit.editingFinished.connect(self._update_sensitivity_summary)
        self.thresholds_edit.editingFinished.connect(self._update_sensitivity_summary)
        self.sobol_samples_spin.valueChanged.connect(lambda _: self._update_sensitivity_summary())
        self.sobol_bootstrap_spin.valueChanged.connect(lambda _: self._update_sensitivity_summary())

        # Visualization controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Quantity of Interest:"))
        self.sens_qoi_combo = QComboBox()
        self.sens_qoi_combo.currentIndexChanged.connect(self._update_sensitivity_plot)
        self.sens_qoi_combo.setToolTip(
            "Select which QoI to display. Options include capacitance at specific horizons "
            "and time-to-failure for various thresholds."
        )
        controls.addWidget(self.sens_qoi_combo, stretch=1)

        self.sens_refresh_btn = QPushButton("Load latest FK Sobol")
        self.sens_refresh_btn.setEnabled(False)
        self.sens_refresh_btn.setToolTip(
            "Pull Sobol indices from the most recent FK forecast. "
            "Enable 'Sobol indices' in the Overview tab and run the forecast first."
        )
        self.sens_refresh_btn.clicked.connect(self._refresh_data_backed_sobol)
        controls.addWidget(self.sens_refresh_btn)

        layout.addLayout(controls)

        # Canvas for Sobol bar charts
        if MATPLOTLIB_AVAILABLE:
            self.sens_fig = Figure(figsize=(10, 8), dpi=300)
            self.sens_fig.set_facecolor("white")
            self.sens_canvas = FigureCanvas(self.sens_fig)
            layout.addWidget(self.sens_canvas, stretch=1)
        else:  # pragma: no cover
            layout.addWidget(QLabel("Matplotlib not available."))

        self.tabs.addTab(sens, "Sensitivity")
        self._update_sensitivity_summary()


    def _build_forecast_tab(self) -> None:
        forecast = QWidget()
        layout = QHBoxLayout(forecast)
        layout.setSpacing(12)

        if MATPLOTLIB_AVAILABLE:
            self.forecast_fig = Figure(figsize=(10, 8), dpi=300)
            self.forecast_fig.set_facecolor("white")
            self.forecast_canvas = FigureCanvas(self.forecast_fig)
            layout.addWidget(self.forecast_canvas, stretch=3)
        else:  # pragma: no cover
            layout.addWidget(QLabel("Matplotlib not available."), stretch=3)

        side = QVBoxLayout()
        band_box = QGroupBox("Bands")
        band_layout = QVBoxLayout(band_box)
        self.epistemic_cb = QCheckBox("Epistemic")
        self.epistemic_cb.setChecked(False)
        self.epistemic_cb.setToolTip("Parameter uncertainty only (advanced)")
        self.epistemic_cb.stateChanged.connect(self._update_forecast_plot)
        band_layout.addWidget(self.epistemic_cb)
        self.total_cb = QCheckBox("Total")
        self.total_cb.setChecked(False)
        self.total_cb.setToolTip("Parameter + observation noise (advanced)")
        self.total_cb.stateChanged.connect(self._update_forecast_plot)
        band_layout.addWidget(self.total_cb)
        self.conformal_cb = QCheckBox("Conformal")
        self.conformal_cb.setChecked(True)
        self.conformal_cb.setToolTip("Distribution-free guaranteed coverage")
        self.conformal_cb.stateChanged.connect(self._update_forecast_plot)
        band_layout.addWidget(self.conformal_cb)
        self.hybrid_cb = QCheckBox("Hybrid")
        self.hybrid_cb.setChecked(True)
        self.hybrid_cb.setToolTip("Conservative envelope (union of intervals)")
        self.hybrid_cb.stateChanged.connect(self._update_forecast_plot)
        band_layout.addWidget(self.hybrid_cb)
        side.addWidget(band_box)

        self.forecast_metrics = QPlainTextEdit()
        self.forecast_metrics.setReadOnly(True)
        self.forecast_metrics.setMaximumWidth(260)
        side.addWidget(self.forecast_metrics, stretch=1)
        side.addStretch(1)
        layout.addLayout(side, stretch=1)

        self.tabs.addTab(forecast, "Forecast & UQ")

    # ------------------------------------------------------------------ Data handling
    def open_file(self) -> None:
        filters = "Data files (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls)"
        file_name, _ = QFileDialog.getOpenFileName(self, "Select dataset", str(Path.cwd()), filters)
        if not file_name:
            return
        try:
            data, time_col, target_col = load_dataset(file_name)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load", str(exc))
            return

        self._data = data
        self._time_column = time_col
        self._target_column = target_col
        self.file_label.setText(f"Loaded {Path(file_name).name} ({len(data)} rows)")

        self.time_combo.clear()
        self.target_combo.clear()
        for column in data.columns:
            self.time_combo.addItem(column)
            self.target_combo.addItem(column)
        self.time_combo.setCurrentText(time_col)
        self.target_combo.setCurrentText(target_col)
        self.time_combo.setEnabled(True)
        self.target_combo.setEnabled(True)
        self.run_button.setEnabled(True)
        self.comparison_btn.setEnabled(True)
        self.save_plot_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self._clear_outputs()

    def _choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select output directory", str(self.output_dir))
        if directory:
            self.output_dir = Path(directory)
            self.output_dir_label.setText(f"Output directory: {self.output_dir}")

    def _model_changed(self) -> None:
        if not hasattr(self, "model_combo"):
            return
        index = self.model_combo.currentIndex()
        self._current_model = self.model_combo.itemData(index)
        self._apply_sensitivity_state()
        self._toggle_mcmc_controls()

    def _mode_changed(self) -> None:
        index = self.mode_combo.currentIndex()
        self.current_mode = self.mode_combo.itemData(index)
        is_forecast = self.current_mode == "forecast"

        if self._active_worker is None:
            self._stop_progress(None)

        self.train_spin.setEnabled(is_forecast)
        self.confidence_spin.setEnabled(is_forecast)
        self.model_combo.setEnabled(is_forecast)
        self._apply_sensitivity_state()
        self.run_button.setEnabled(is_forecast and self._data is not None)
        if hasattr(self, "comparison_btn"):
            self.comparison_btn.setEnabled(is_forecast and self._data is not None)
        self.save_plot_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        if MATPLOTLIB_AVAILABLE and hasattr(self, "save_sensitivity_btn"):
            self.save_sensitivity_btn.setEnabled(False)
        if hasattr(self, "sens_run_button"):
            self.sens_run_button.setEnabled(self.current_mode == "sensitivity" and self._active_worker is None)
        if MATPLOTLIB_AVAILABLE and hasattr(self, "sens_save_plot_btn"):
            self.sens_save_plot_btn.setEnabled(False)

        if hasattr(self, "tabs") and self.tabs is not None:
            if self.current_mode == "sensitivity":
                target = getattr(self, "sensitivity_tab", None)
                if target is not None:
                    self.tabs.setCurrentWidget(target)
            elif self.current_mode == "forecast":
                self.tabs.setCurrentIndex(0)

        if self._active_worker is None:
            self._stop_progress("Ready")
    def _apply_sensitivity_state(self) -> None:
        is_forecast = self.current_mode == "forecast"
        is_fk = hasattr(self, "model_combo") and self.model_combo.currentData() == "fk"
        allow = is_forecast and is_fk
        if hasattr(self, "sensitivity_check"):
            if not allow and self.sensitivity_check.isChecked():
                self.sensitivity_check.blockSignals(True)
                self.sensitivity_check.setChecked(False)
                self.sensitivity_check.blockSignals(False)
            self.sensitivity_check.setEnabled(allow)
        if hasattr(self, "sensitivity_config_btn"):
            self.sensitivity_config_btn.setEnabled(is_fk)
        self._update_sensitivity_summary()

    def _open_sensitivity_tab(self) -> None:
        if hasattr(self, "tabs") and getattr(self, "sensitivity_tab", None) is not None:
            self.tabs.setCurrentWidget(self.sensitivity_tab)

    def _format_sensitivity_summary(self) -> str:
        horizons_widget = getattr(self, "sobol_horizons_edit", None)
        thresholds_widget = getattr(self, "thresholds_edit", None)
        samples_widget = getattr(self, "sobol_samples_spin", None)
        bootstrap_widget = getattr(self, "sobol_bootstrap_spin", None)
        horizons = horizons_widget.text().strip() if horizons_widget is not None else ""
        thresholds = thresholds_widget.text().strip() if thresholds_widget is not None else ""
        samples = str(samples_widget.value()) if samples_widget is not None else ""
        bootstrap = str(bootstrap_widget.value()) if bootstrap_widget is not None else ""
        return f"H: {horizons or 'n/a'}  |  q: {thresholds or 'n/a'}  |  N: {samples or 'n/a'}  |  B: {bootstrap or 'n/a'}"

    def _update_sensitivity_summary(self) -> None:
        if hasattr(self, "sensitivity_summary"):
            self.sensitivity_summary.setText(self._format_sensitivity_summary())

    def _refresh_data_backed_sobol(self) -> None:
        if not self._last_result:
            QMessageBox.information(
                self,
                "No forecast available",
                "Run the FK forecast with Sobol indices enabled to populate data-backed results.",
            )
            return
        self._update_sensitivity_tab(self._last_result)
        QMessageBox.information(self, "Sobol refreshed", "Loaded Sobol indices from the latest FK forecast.")

    def _toggle_mcmc_controls(self) -> None:
        enabled = self.mcmc_check.isChecked()
        self.mcmc_draws_spin.setEnabled(enabled)
        self.mcmc_burnin_spin.setEnabled(enabled)
        self.mcmc_step_spin.setEnabled(enabled)

    def _generate_baseline_comparison(self) -> None:
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Comparison plots", "Matplotlib backend not available; cannot generate plots.")
            return
        if self._data is None or self._time_column is None or self._target_column is None:
            QMessageBox.warning(self, "Comparison plots", "Load a dataset before generating comparison plots.")
            return

        def _safe_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float('nan')

        try:
            self._start_progress("Generating baseline comparison plots...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            frame = (
                self._data[[self._time_column, self._target_column]]
                .dropna()
                .sort_values(self._time_column)
            )
            times_all = frame[self._time_column].to_numpy(dtype=float)
            values_all = frame[self._target_column].to_numpy(dtype=float)
            picp = PICPCore(use_legacy=False, fallback_to_modern=True)
            models = [
                ("classical", "Classical mass balance", "universal_picp_Target_forecast.png"),
                ("donor", "Donor Prony-2", "universal_picp_Target_forecast_DP2.png"),
                ("kww", "KWW stretched exponential", "universal_picp_Target_forecast_FK.png"),
            ]
            for model_type, label, filename in models:
                result = picp.run_forecast(
                    frame,
                    time_column=self._time_column,
                    target_column=self._target_column,
                    confidence=float(self.confidence_spin.value()),
                    train_ratio=float(self.train_spin.value()),
                    model_type=model_type,
                )
                split = result["pipeline"]["split"]
                fit_count = split["fit_count"]
                cal_count = split["calibration_count"]
                train_count = fit_count + cal_count
                mu_fit = np.asarray(result["pipeline"]["model_fitting"]["mu_hat_fit"], dtype=float)
                mu_cal = np.asarray(result["pipeline"]["model_fitting"]["mu_hat_cal"], dtype=float)
                mu_fore = np.asarray(result["pipeline"]["prediction"]["mu_hat_forecast"], dtype=float)
                pred_time = np.concatenate(
                    [
                        times_all[:fit_count],
                        times_all[fit_count:train_count],
                        times_all[train_count : train_count + mu_fore.size],
                    ]
                )
                mean_curve = np.concatenate([mu_fit, mu_cal, mu_fore])
                forecast_df = pd.DataFrame(result["forecast_segment"])
                times_fore = forecast_df["time"].to_numpy(dtype=float)
                conf_low = forecast_df["lower"].to_numpy(dtype=float)
                conf_high = forecast_df["upper"].to_numpy(dtype=float)
                obs_fore = forecast_df["observed"].to_numpy(dtype=float)
                fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
                ax.set_xlim(0, 564)
                ax.set_ylim(30.0, 100.0)
                legend_entries: list[tuple[Any, str]] = []
                seen: set[str] = set()

                def _legend(handle: Any, text: str) -> None:
                    if handle is not None and text not in seen:
                        legend_entries.append((handle, text))
                        seen.add(text)
                train_handle = ax.scatter(
                    times_all[:train_count],
                    values_all[:train_count],
                    color=TRAIN_COLOR,
                    edgecolors="white",
                    linewidths=0.5,
                    s=22,
                    alpha=0.85,
                )
                _legend(train_handle, "Training data")
                obs_handle = ax.scatter(
                    times_all[train_count : train_count + obs_fore.size],
                    obs_fore,
                    color=FORECAST_OBS_COLOR,
                    edgecolors="white",
                    linewidths=0.6,
                    s=28,
                    alpha=0.9,
                )
                _legend(obs_handle, "Observed forecast")
                mean_handle, = ax.plot(pred_time, mean_curve, color=FORECAST_MEAN_COLOR, linewidth=2.1)
                _legend(mean_handle, "Model mean")
                if train_count > 0:
                    boundary_handle = ax.axvline(
                        times_all[train_count - 1],
                        color="#8a857d",
                        linestyle="--",
                        linewidth=1.1,
                    )
                    _legend(boundary_handle, "Forecast boundary")
                mask = np.isfinite(conf_low) & np.isfinite(conf_high)
                if mask.any():
                    conf_handle = ax.fill_between(
                        times_fore[mask],
                        conf_low[mask],
                        conf_high[mask],
                        color=CONFORMAL_COLOR,
                        alpha=0.22,
                    )
                    _legend(conf_handle, "Conformal interval (guaranteed)")
                    ax.plot(times_fore[mask], conf_low[mask], color=CONFORMAL_COLOR, linewidth=1.2)
                    ax.plot(times_fore[mask], conf_high[mask], color=CONFORMAL_COLOR, linewidth=1.2)
                metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
                rmse_val = _safe_float(metrics.get("rmse_forecast"))
                mae_val = _safe_float(metrics.get("mae_forecast"))
                coverage_val = _safe_float(metrics.get("coverage_forecast"))
                metrics_box = []
                if np.isfinite(rmse_val):
                    metrics_box.append(f"RMSE: {rmse_val:.3f}")
                if np.isfinite(mae_val):
                    metrics_box.append(f"MAE: {mae_val:.3f}")
                if np.isfinite(coverage_val):
                    metrics_box.append(f"Coverage: {coverage_val:.3f}")
                if metrics_box:
                    ax.text(
                        0.72,
                        0.5,
                        "\n".join(metrics_box),
                        transform=ax.transAxes,
                        ha="left",
                        va="center",
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.75, edgecolor="#888888"),
                    )
                target_label = self._target_column or "Capacitance"
                ax.set_title(f"{label} forecast for {target_label}")
                ax.set_xlabel(self._time_column or "Time")
                ylabel = target_label
                if isinstance(ylabel, str) and ylabel.lower().startswith("cap"):
                    ylabel = f"{ylabel} (uF)"
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)
                if legend_entries:
                    handles, labels_txt = zip(*legend_entries)
                    ax.legend(handles, labels_txt, loc="upper right", frameon=True, framealpha=0.85, fontsize=11, ncol=2)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(labelsize=10)
                fig.tight_layout()
                output_path = self.output_dir / filename
                fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
                plt.close(fig)
            QMessageBox.information(self, "Comparison plots", f"Saved baseline plots to {self.output_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Comparison plots", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            self._stop_progress("Ready")

    # ------------------------------------------------------------------ Analysis
    def run_analysis(self) -> None:
        if self.current_mode == "sensitivity":
            target = getattr(self, "sensitivity_tab", None)
            if target is not None:
                self.tabs.setCurrentWidget(target)
            QMessageBox.information(self, "Sensitivity study", "Use the Sensitivity tab controls to configure and run the study.")
            return
        if self._active_worker is not None:
            QMessageBox.information(self, "Analysis running", "Please wait for the current analysis to complete.")
            return
        if self._data is None or self._time_column is None or self._target_column is None:
            QMessageBox.warning(self, "No data", "Please load a dataset first.")
            return

        self._reset_pending_inputs()

        subset = self._data[[self.time_combo.currentText(), self.target_combo.currentText()]].dropna()
        if subset.empty:
            QMessageBox.warning(self, "No data", "Dataset is empty after removing missing rows.")
            return
        subset = subset.copy().reset_index(drop=True)

        time_col = subset.columns[0]
        target_col = subset.columns[1]
        times = subset.iloc[:, 0].to_numpy(dtype=float)
        values = subset.iloc[:, 1].to_numpy(dtype=float)

        self._pending_model = self._current_model
        self._pending_times = times.copy()
        self._pending_values = values.copy()
        self._last_result = None
        self._current_forecast = None
        self.sens_data = None
        self.forecast_data = None
        self.save_plot_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self._time_values = None
        self._value_values = None

        model_key = self._current_model
        train_ratio = float(self.train_spin.value())

        if model_key == "fk":
            try:
                thresholds = self._parse_float_list(
                    self.thresholds_edit.text(),
                    min_value=0.0,
                    max_value=1.0,
                    field_name="failure threshold",
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid threshold settings", str(exc))
                self._reset_pending_inputs()
                return

            try:
                horizons = self._parse_float_list(
                    self.sobol_horizons_edit.text(),
                    min_value=0.0,
                    field_name="sensitivity horizon",
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid sensitivity settings", str(exc))
                self._reset_pending_inputs()
                return

            config = FractionalConfig(
                train_ratio=train_ratio,
                confidence=float(self.confidence_spin.value()),
                run_sensitivity=self.sensitivity_check.isChecked(),
                bootstrap_draws=int(self.bootstrap_spin.value()),
                sobol_samples=int(self.sobol_samples_spin.value()),
                sobol_bootstrap=int(self.sobol_bootstrap_spin.value()),
                use_mcmc=self.mcmc_check.isChecked(),
                mcmc_draws=int(self.mcmc_draws_spin.value()),
                mcmc_burn_in=int(self.mcmc_burnin_spin.value()),
                mcmc_step_scale=float(self.mcmc_step_spin.value()),
                thresholds=tuple(thresholds),
                sensitivity_horizons=tuple(horizons),
            )

            self._start_progress("Running FK forecast...")
            self.run_button.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)

            worker = AnalysisWorker(
                self._run_fk_task,
                subset,
                config,
                time_col,
                target_col,
            )
            worker.signals.finished.connect(self._on_analysis_success)
        else:
            label = "Classical forecast..." if model_key == "classical" else "KWW forecast..."
            self._start_progress(label)
            self.run_button.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            worker_fn = self._run_classical_task if model_key == "classical" else self._run_kww_task
            worker = AnalysisWorker(worker_fn, times.copy(), values.copy(), train_ratio)
            worker.signals.finished.connect(self._on_surrogate_success)

        worker.signals.error.connect(self._on_analysis_error)
        self._active_worker = worker
        self.thread_pool.start(worker)

    @staticmethod
    def _run_fk_task(
        data: pd.DataFrame,
        config: FractionalConfig,
        time_col: str,
        target_col: str,
    ) -> Dict[str, Any]:
        core = FractionalPICPCore(config)
        return core.run_forecast(
            data,
            time_column=time_col,
            target_column=target_col,
        )

    @staticmethod
    def _run_classical_task(
        times: np.ndarray,
        values: np.ndarray,
        train_ratio: float,
    ) -> Dict[str, Any]:
        return fit_classical(times, values, train_ratio=train_ratio)

    @staticmethod
    def _run_kww_task(
        times: np.ndarray,
        values: np.ndarray,
        train_ratio: float,
    ) -> Dict[str, Any]:
        return fit_kww(times, values, train_ratio=train_ratio)

    def _on_analysis_success(self, result: Dict[str, Any]) -> None:
        QApplication.restoreOverrideCursor()
        self.run_button.setEnabled(True)
        self._active_worker = None
        self._stop_progress("Analysis complete.")
        self._last_result = result

        data_info = result.get("data", {})
        if isinstance(data_info, dict):
            self._time_values = np.asarray(data_info.get("times", []), dtype=float)
            self._value_values = np.asarray(data_info.get("values", []), dtype=float)
        else:
            self._time_values = self._pending_times.copy() if self._pending_times is not None else None
            self._value_values = self._pending_values.copy() if self._pending_values is not None else None

        self._update_overview_tab(result)
        self._update_sensitivity_tab(result)
        self._update_forecast_tab(result)
        self.save_plot_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        QMessageBox.information(self, "Forecast complete", "Prediction and uncertainty bands are ready.")
        self._reset_pending_inputs()

    def _on_surrogate_success(self, result: Dict[str, Any]) -> None:
        QApplication.restoreOverrideCursor()
        self.run_button.setEnabled(True)
        self._active_worker = None
        self._stop_progress("Analysis complete.")

        if not isinstance(result, dict):
            QMessageBox.information(self, "Forecast complete", "Prediction ready.")
            self._reset_pending_inputs()
            return

        model_label = "Classical exponential" if self._pending_model == "classical" else "KWW stretched exponential"
        title = model_label
        self._time_values = self._pending_times.copy() if self._pending_times is not None else None
        self._value_values = self._pending_values.copy() if self._pending_values is not None else None
        self._render_surrogate(title, result)
        self._current_forecast = {
            "time": np.asarray(result.get("time", []), dtype=float).tolist(),
            "prediction": np.asarray(result.get("prediction", []), dtype=float).tolist(),
        }
        self._last_result = {
            "forecast": self._current_forecast,
            "metrics": result.get("metrics", {}),
        }
        self.save_plot_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        QMessageBox.information(self, "Forecast complete", "Prediction ready.")
        self._reset_pending_inputs()

    def _render_surrogate(self, title: str, result: Dict[str, Any]) -> None:
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, "forecast_fig"):
            return

        self.forecast_data = None
        times = np.asarray(result.get("time", []), dtype=float)
        prediction = np.asarray(result.get("prediction", []), dtype=float)
        train_idx = int(result.get("train_idx", 0) or 0)
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}

        self.forecast_fig.clf()
        ax = self.forecast_fig.add_subplot(111)

        legend_entries: list[tuple[Any, str]] = []
        seen: set[str] = set()

        def _legend(handle: Any, label: str) -> None:
            if handle is not None and label not in seen:
                legend_entries.append((handle, label))
                seen.add(label)

        if self._time_values is not None and self._value_values is not None:
            total_points = min(len(self._time_values), len(self._value_values))
            split = min(train_idx, total_points)
            if split:
                train_handle = ax.scatter(
                    self._time_values[:split],
                    self._value_values[:split],
                    color=TRAIN_COLOR,
                    edgecolors="#ffffff",
                    linewidths=0.5,
                    s=20,
                )
                _legend(train_handle, "Training data")
            if split < total_points:
                obs_handle = ax.scatter(
                    self._time_values[split:total_points],
                    self._value_values[split:total_points],
                    color=FORECAST_OBS_COLOR,
                    edgecolors="#ffffff",
                    linewidths=0.6,
                    s=28,
                )
                _legend(obs_handle, "Observed forecast")

        if times.size and prediction.size:
            pred_handle, = ax.plot(times, prediction, color=FORECAST_MEAN_COLOR, linewidth=2.0)
            _legend(pred_handle, "Model prediction")

        if train_idx > 0 and train_idx <= times.size:
            boundary_handle = ax.axvline(
                times[train_idx - 1],
                color="#8a857d",
                linestyle="--",
                linewidth=1.1,
            )
            _legend(boundary_handle, "Forecast boundary")

        ax.set_title(title)
        ax.set_xlabel(self._time_column or "Time")
        ax.set_ylabel(self._target_column or "Response")
        ax.set_xlim(0, 564)
        ax.set_ylim(30.0, 100.0)
        ax.grid(True, alpha=0.25)
        if legend_entries:
            handles, labels = zip(*legend_entries)
            ax.legend(handles, labels, loc="upper right", frameon=True, framealpha=0.85, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        self.forecast_fig.tight_layout()
        self.forecast_canvas.draw()

        metric_lines = ["Surrogate metrics:"]
        for label, key in (
            ("RMSE (train)", "rmse_train"),
            ("MAPE (train)", "mape_train"),
            ("RMSE (forecast)", "rmse_forecast"),
            ("MAPE (forecast)", "mape_forecast"),
        ):
            if key in metrics and metrics[key] is not None:
                metric_lines.append(f"  {label}: {self._fmt(metrics[key])}")
        if len(metric_lines) == 1:
            metric_lines.append("  (metrics unavailable)")
        self.forecast_metrics.setPlainText("\n".join(metric_lines))

    def _on_analysis_error(self, message: str) -> None:
        QApplication.restoreOverrideCursor()
        self._stop_progress("Ready")
        self._active_worker = None
        self._current_forecast = None
        self._last_result = None
        self._reset_pending_inputs()
        QMessageBox.critical(self, "Analysis failed", message or "Unknown error during analysis.")

    def _reset_pending_inputs(self) -> None:
        self._pending_model = None
        self._pending_times = None
        self._pending_values = None

    def _start_progress(self, message: str) -> None:
        self.status_label.setText(message)
        if self.progress_bar is not None:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.show()

    def _stop_progress(self, message: Optional[str] = None) -> None:
        if message is not None:
            self.status_label.setText(message)
        if self.progress_bar is not None:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_bar.hide()

    @staticmethod
    def _parse_float_list(
        text: str,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_empty: bool = False,
        field_name: str = "value",
    ) -> list[float]:
        values: list[float] = []
        for raw in text.split(','):
            token = raw.strip()
            if not token:
                continue
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError(f"{field_name.title()} entries must be numeric.") from exc
            if min_value is not None and not (value > min_value):
                raise ValueError(f"Each {field_name} must be greater than {min_value}.")
            if max_value is not None and not (value < max_value):
                raise ValueError(f"Each {field_name} must be less than {max_value}.")
            values.append(value)
        if not values and not allow_empty:
            raise ValueError(f"Enter at least one {field_name}.")
        return values

    # ------------------------------------------------------------------ Overview tab
    def _update_overview_tab(self, result: Dict[str, Any]) -> None:
        params = result.get("fit", {}).get("params", {})
        self.param_table.setRowCount(len(params))
        for row, (name, value) in enumerate(params.items()):
            self.param_table.setItem(row, 0, QTableWidgetItem(name))
            self.param_table.setItem(row, 1, QTableWidgetItem(self._fmt(value)))
        self.param_table.resizeColumnsToContents()

        metrics = result.get("metrics", {})
        self.metric_table.setRowCount(len(metrics))
        for row, (name, value) in enumerate(metrics.items()):
            self.metric_table.setItem(row, 0, QTableWidgetItem(name))
            self.metric_table.setItem(row, 1, QTableWidgetItem(self._fmt(value)))
        self.metric_table.resizeColumnsToContents()

        data_info = result.get("data", {})
        fit_info = result.get("fit", {})
        bias_info = result.get("bias_correction")
        posterior_info = result.get("posterior", {})
        mcmc_info = posterior_info.get("mcmc") if isinstance(posterior_info, dict) else None
        lines = [
            f"Training points: {data_info.get('train_count', 'n/a')}",
            f"Calibration points: {data_info.get('calibration_count', 'n/a')}",
            f"Forecast points: {data_info.get('forecast_count', 'n/a')}",
            "",
            "Diagnostics:",
            f"  Shapiro p-value: {self._fmt(metrics.get('residual_shapiro_p'))}",
            f"  Runs p-value: {self._fmt(metrics.get('residual_runs_p'))}",
            f"  WAIC: {self._fmt(metrics.get('WAIC'))}",
            f"  sigma_log: {self._fmt(fit_info.get('sigma_log'))}",
            f"  Hessian cond.: {self._fmt(fit_info.get('hessian_condition'))}",
            f"  Monotonic (train): {self._fmt(fit_info.get('monotonic'))}",
        ]
        if isinstance(bias_info, dict) and bias_info:
            if "bias_factor" in bias_info:
                lines.append(f"  Bias factor: {self._fmt(bias_info.get('bias_factor'))}")
            if "ci_low" in bias_info and "ci_high" in bias_info:
                ci_low = self._fmt(bias_info.get("ci_low"))
                ci_high = self._fmt(bias_info.get("ci_high"))
                lines.append(f"  Bias factor CI: [{ci_low}, {ci_high}]")
        if isinstance(mcmc_info, dict):
            if "acceptance_rate" in mcmc_info:
                lines.append(f"  MCMC acceptance: {self._fmt(mcmc_info.get('acceptance_rate'))}")
            if "error" in mcmc_info:
                lines.append(f"  MCMC status: {mcmc_info['error']}")
        self.diagnostics_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ Sensitivity tab
    def _update_sensitivity_tab(self, result: Dict[str, Any]) -> None:
        self.sens_data = result.get("sensitivity")
        self.sens_qoi_combo.blockSignals(True)
        self.sens_qoi_combo.clear()
        if not self.sens_data:
            self.sens_qoi_combo.addItem("Sensitivity not computed")
            self.sens_qoi_combo.setEnabled(False)
            if MATPLOTLIB_AVAILABLE:
                self.sens_fig.clf()
                self.sens_canvas.draw()
            if MATPLOTLIB_AVAILABLE and hasattr(self, "save_sensitivity_btn"):
                self.save_sensitivity_btn.setEnabled(False)
            if hasattr(self, "sens_refresh_btn"):
                self.sens_refresh_btn.setEnabled(False)
            self.sens_qoi_combo.blockSignals(False)
            return
        for key in sorted(self.sens_data.keys()):
            self.sens_qoi_combo.addItem(key)
        self.sens_qoi_combo.setEnabled(True)
        self.sens_qoi_combo.blockSignals(False)
        if MATPLOTLIB_AVAILABLE and hasattr(self, "save_sensitivity_btn"):
            has_valid = any(isinstance(val, dict) and "S" in val for val in self.sens_data.values())
            self.save_sensitivity_btn.setEnabled(has_valid)
        if hasattr(self, "sens_refresh_btn"):
            self.sens_refresh_btn.setEnabled(True)
        self._update_sensitivity_plot()

    def _update_sensitivity_plot(self) -> None:
        """
        Render Sobol indices using the unified _draw_sobol_panel helper.

        Mode-aware: displays forecast-backed results (priors anchored at MAP fit)
        or prior-sweep results (data-free parameter space exploration).
        """
        if not MATPLOTLIB_AVAILABLE or not getattr(self, "sens_data", None):
            return

        key = self.sens_qoi_combo.currentText()
        data = self.sens_data.get(key)
        if not data:
            self.sens_fig.clf()
            self.sens_canvas.draw()
            return

        # Handle error case
        if isinstance(data, dict) and "error" in data and not {"S", "S_total"} <= set(data.keys()):
            self.sens_fig.clf()
            ax_msg = self.sens_fig.add_subplot(111)
            ax_msg.axis("off")
            ax_msg.text(
                0.5,
                0.5,
                data.get("error", "Sensitivity analysis failed."),
                ha="center",
                va="center",
                wrap=True,
                fontsize=10,
            )
            self.sens_canvas.draw()
            if hasattr(self, "save_sensitivity_btn"):
                self.save_sensitivity_btn.setEnabled(False)
            return

        # Extract Sobol data
        params = ["C0", "k", "alpha", "f_inf"]
        S = np.asarray(data["S"], dtype=float)
        ST = np.asarray(data["S_total"], dtype=float)
        S_ci = np.asarray(data.get("S_ci"), dtype=float) if data.get("S_ci") is not None else None
        ST_ci = np.asarray(data.get("S_total_ci"), dtype=float) if data.get("S_total_ci") is not None else None

        # Draw using unified helper
        self.sens_fig.clf()
        ax = self.sens_fig.add_subplot(111)
        _draw_sobol_panel(ax, S, ST, params, S_ci, ST_ci, title=key)

        self.sens_fig.tight_layout()
        if hasattr(self, "save_sensitivity_btn"):
            self.save_sensitivity_btn.setEnabled(True)
        self.sens_canvas.draw()

    # ------------------------------------------------------------------ Forecast tab
    def _update_forecast_tab(self, result: Dict[str, Any]) -> None:
        if MATPLOTLIB_AVAILABLE:
            self.forecast_data = result
            self._update_forecast_plot()
            if hasattr(self, "save_sensitivity_btn") and self.sens_data:
                self.save_sensitivity_btn.setEnabled(True)

    def _update_forecast_plot(self) -> None:
        if not MATPLOTLIB_AVAILABLE or not getattr(self, "forecast_data", None):
            return
        result = self.forecast_data
        forecast = result.get("forecast", {})
        times = np.asarray(forecast.get("time", []), dtype=float)
        mean_curve = np.asarray(forecast.get("mean", []), dtype=float)
        bias_corrected_raw = forecast.get("mean_bias_corrected")
        mean_bias_corrected = (
            np.asarray(bias_corrected_raw, dtype=float)
            if bias_corrected_raw is not None
            else np.array([], dtype=float)
        )
        ep_low = np.asarray(forecast.get("epistemic_low", []), dtype=float)
        ep_high = np.asarray(forecast.get("epistemic_high", []), dtype=float)
        tot_low = np.asarray(forecast.get("total_low", []), dtype=float)
        tot_high = np.asarray(forecast.get("total_high", []), dtype=float)
        conf_low = np.asarray(forecast.get("conformal_low", []), dtype=float)
        conf_high = np.asarray(forecast.get("conformal_high", []), dtype=float)
        hybrid_low = np.asarray(forecast.get("hybrid_low", []), dtype=float)
        hybrid_high = np.asarray(forecast.get("hybrid_high", []), dtype=float)

        data_info = result.get("data", {})
        train_count = data_info.get("train_count", 0)
        forecast_count = data_info.get("forecast_count", 0)

        self.forecast_fig.clf()
        ax = self.forecast_fig.add_subplot(111)

        legend_entries: list[tuple[Any, str]] = []
        seen: set[str] = set()

        def _legend(handle: Any, label: str) -> None:
            if handle is not None and label not in seen:
                legend_entries.append((handle, label))
                seen.add(label)

        if self._time_values is not None and self._value_values is not None:
            if train_count:
                train_handle = ax.scatter(
                    self._time_values[:train_count],
                    self._value_values[:train_count],
                    color=TRAIN_COLOR,
                    edgecolors="#ffffff",
                    linewidths=0.5,
                    s=20,
                )
                _legend(train_handle, "Training data")
            if forecast_count:
                start = train_count
                end = train_count + forecast_count
                obs_handle = ax.scatter(
                    self._time_values[start:end],
                    self._value_values[start:end],
                    color=FORECAST_OBS_COLOR,
                    edgecolors="#ffffff",
                    linewidths=0.6,
                    s=28,
                )
                _legend(obs_handle, "Observed forecast")
        if mean_curve.size:
            mean_handle, = ax.plot(times, mean_curve, color=FORECAST_MEAN_COLOR, linewidth=2.2)
            _legend(mean_handle, "Forecast mean")
        if mean_bias_corrected.size:
            bias_handle, = ax.plot(
                times,
                mean_bias_corrected,
                color=BIAS_MEAN_COLOR,
                linestyle="--",
                linewidth=1.7,
            )
            _legend(bias_handle, "Bias-corrected mean")
        if train_count and train_count <= times.size:
            boundary_handle = ax.axvline(
                times[train_count - 1],
                color="#8a857d",
                linestyle="--",
                linewidth=1.2,
            )
            _legend(boundary_handle, "Forecast boundary")

        if self.epistemic_cb.isChecked() and np.isfinite(ep_low).any():
            mask = np.isfinite(ep_low) & np.isfinite(ep_high)
            if mask.any():
                epi_handle = ax.fill_between(
                    times[mask],
                    ep_low[mask],
                    ep_high[mask],
                    color=EPISTEMIC_COLOR,
                    alpha=0.18,
                )
                _legend(epi_handle, "Epistemic")
                ax.plot(times[mask], ep_low[mask], color=EPISTEMIC_COLOR, linewidth=1.0, linestyle="--", alpha=0.9)
                ax.plot(times[mask], ep_high[mask], color=EPISTEMIC_COLOR, linewidth=1.0, linestyle="--", alpha=0.9)
        if self.total_cb.isChecked() and np.isfinite(tot_low).any():
            mask = np.isfinite(tot_low) & np.isfinite(tot_high)
            if mask.any():
                total_handle = ax.fill_between(
                    times[mask],
                    tot_low[mask],
                    tot_high[mask],
                    color=TOTAL_COLOR,
                    alpha=0.18,
                )
                _legend(total_handle, "Total")
                ax.plot(times[mask], tot_low[mask], color=TOTAL_COLOR, linewidth=1.0, linestyle="-.", alpha=0.85)
                ax.plot(times[mask], tot_high[mask], color=TOTAL_COLOR, linewidth=1.0, linestyle="-.", alpha=0.85)
        if self.conformal_cb.isChecked() and np.isfinite(conf_low).any():
            mask = np.isfinite(conf_low) & np.isfinite(conf_high)
            if mask.any():
                conf_handle = ax.fill_between(
                    times[mask],
                    conf_low[mask],
                    conf_high[mask],
                    color=CONFORMAL_COLOR,
                    alpha=0.22,
                )
                _legend(conf_handle, "Conformal")
                ax.plot(times[mask], conf_low[mask], color="#b89560", linewidth=1.2, linestyle="-")
                ax.plot(times[mask], conf_high[mask], color="#b89560", linewidth=1.2, linestyle="-")
        if self.hybrid_cb.isChecked() and np.isfinite(hybrid_low).any():
            mask = np.isfinite(hybrid_low) & np.isfinite(hybrid_high)
            if mask.any():
                hybrid_handle = ax.fill_between(
                    times[mask],
                    hybrid_low[mask],
                    hybrid_high[mask],
                    color=HYBRID_COLOR,
                    alpha=0.18,
                )
                _legend(hybrid_handle, "Hybrid")
                ax.plot(times[mask], hybrid_low[mask], color=HYBRID_COLOR, linewidth=1.3, linestyle="-")
                ax.plot(times[mask], hybrid_high[mask], color=HYBRID_COLOR, linewidth=1.3, linestyle="-")

        ax.set_xlabel(self._time_column or "Time")
        ylabel = self._target_column or "Capacitance"
        if isinstance(ylabel, str) and ylabel.lower().startswith("cap"):
            ylabel = f"{ylabel} (uF)"
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 564)
        ax.set_ylim(30.0, 100.0)
        ax.grid(True, alpha=0.25)
        if legend_entries:
            handles, labels = zip(*legend_entries)
            ax.legend(handles, labels, loc="upper right", frameon=True, framealpha=0.85, fontsize=11, ncol=2, columnspacing=1.2)
        metrics = result.get("metrics", {})
        summary_lines = []
        rmse_val = metrics.get("rmse_forecast")
        mae_val = metrics.get("mae_forecast")
        cov_val = metrics.get("conformal_coverage")
        if rmse_val is not None:
            summary_lines.append(f"RMSE: {rmse_val:.3f}")
        if mae_val is not None:
            summary_lines.append(f"MAE: {mae_val:.3f}")
        if cov_val is not None:
            summary_lines.append(f"Coverage: {cov_val:.3f}")
        if summary_lines:
            ax.text(
                0.72,
                0.5,
                "\n".join(summary_lines),
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.75, edgecolor="#888888"),
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        self.forecast_fig.tight_layout()
        self.forecast_canvas.draw()

        metrics = result.get("metrics", {})
        posterior = result.get("posterior", {})
        mcmc_info = posterior.get("mcmc") if isinstance(posterior, dict) else None
        bias_info = result.get("bias_correction")
        lines = ["Forecast metrics:"]
        for key in ("rmse_forecast", "mae_forecast", "conformal_coverage"):
            if key in metrics:
                lines.append(f"  {key}: {self._fmt(metrics[key])}")
        if isinstance(bias_info, dict) and "bias_factor" in bias_info:
            lines.append(f"  Bias factor: {self._fmt(bias_info.get('bias_factor'))}")
        if "bias_factor" in metrics:
            lines.append(f"  bias_factor (metrics): {self._fmt(metrics.get('bias_factor'))}")
        if isinstance(mcmc_info, dict):
            if "acceptance_rate" in mcmc_info:
                lines.append(f"  MCMC acceptance: {self._fmt(mcmc_info.get('acceptance_rate'))}")
            if "error" in mcmc_info:
                lines.append(f"  MCMC status: {mcmc_info['error']}")
        tldr = (
            "TL;DR: Epistemic = parameter uncertainty only. Total = parameters + noise. "
            "Conformal = distribution-free guaranteed coverage. Hybrid = conservative envelope (widest)."
        )
        lines.insert(0, tldr)
        self.forecast_metrics.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ Helpers
    def _clear_outputs(self) -> None:
        self.param_table.setRowCount(0)
        self.metric_table.setRowCount(0)
        self.diagnostics_text.clear()
        self.sens_data = None
        self.sens_qoi_combo.clear()
        self.sens_qoi_combo.addItem("Sensitivity not computed")
        self.sens_qoi_combo.setEnabled(False)
        if MATPLOTLIB_AVAILABLE:
            self.sens_fig.clf()
            self.sens_canvas.draw()
            self.forecast_fig.clf()
            self.forecast_canvas.draw()
        if MATPLOTLIB_AVAILABLE and hasattr(self, "save_sensitivity_btn"):
            self.save_sensitivity_btn.setEnabled(False)
        if hasattr(self, "sens_refresh_btn"):
            self.sens_refresh_btn.setEnabled(False)
        self.comparison_btn.setEnabled(False)
        self.forecast_metrics.clear()
        self._last_result = None

    def _save_plot(self) -> None:
        if not (MATPLOTLIB_AVAILABLE and getattr(self, "forecast_fig", None) and self._last_result):
            QMessageBox.warning(self, "Save plot", "Run the analysis before saving a plot.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save forecast plot",
            str((self.output_dir / "fk_forecast").with_suffix(".png")),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        try:
            self.forecast_fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        except Exception as exc:
            QMessageBox.critical(self, "Save plot", str(exc))

    def _save_sensitivity_plot(self) -> None:
        if not (
            MATPLOTLIB_AVAILABLE
            and getattr(self, "sens_fig", None)
            and getattr(self, "sens_data", None)
            and self.sens_data
        ):
            QMessageBox.warning(self, "Save sensitivity plot", "Run sensitivity analysis before saving.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save sensitivity plot",
            str((self.output_dir / "fk_sensitivity").with_suffix(".png")),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        try:
            self.sens_fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        except Exception as exc:
            QMessageBox.critical(self, "Save sensitivity plot", str(exc))

    def _export_forecast(self) -> None:
        if not self._last_result:
            QMessageBox.warning(self, "Export forecast", "Run the analysis before exporting.")
            return
        forecast = self._last_result.get("forecast", {})
        if not forecast:
            QMessageBox.warning(self, "Export forecast", "No forecast data available to export.")
            return
        df = pd.DataFrame(forecast)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export forecast data",
            str((self.output_dir / "fk_forecast").with_suffix(".csv")),
            "CSV (*.csv)",
        )
        if not path:
            return
        try:
            df.to_csv(path, index=False)
        except Exception as exc:
            QMessageBox.critical(self, "Export forecast", str(exc))

    def _apply_palette(self) -> None:
        palette = self.palette()
        palette.setColor(palette.Window, QColor("#f0f0f0"))
        palette.setColor(palette.Base, QColor("#ffffff"))
        palette.setColor(palette.AlternateBase, QColor("#e6e6e6"))
        palette.setColor(palette.Button, QColor("#e0e0e0"))
        palette.setColor(palette.ButtonText, QColor("#202020"))
        palette.setColor(palette.WindowText, QColor("#202020"))
        palette.setColor(palette.Text, QColor("#202020"))
        palette.setColor(palette.Highlight, QColor("#5f6f7d"))
        palette.setColor(palette.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)

    @staticmethod
    def _fmt(value: Optional[Any]) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FractionalWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
