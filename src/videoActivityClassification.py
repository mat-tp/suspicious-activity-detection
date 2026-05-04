import os
import sys
import pickle
import threading
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_PLUGIN_PATH"]  = ""
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

UI_DIR      = Path(__file__).resolve().parent
SRC_DIR     = UI_DIR.parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QGroupBox, QFormLayout, QFileDialog,
    QMessageBox, QStatusBar, QTextEdit, QTabWidget, QScrollArea,
    QComboBox, QProgressBar
)

import settings as cfg
from vision.pipeline import process_video
from ml.features     import extract_all_features, aggregate_video_features, FEATURE_NAMES
from ml.classifier   import ActivityClassifier, build_sklearn_pipelines, build_all_pipelines

class TrainingWorker(QThread):
    log_line  = pyqtSignal(str)
    finished  = pyqtSignal(object, object)
    error     = pyqtSignal(str)

    def __init__(self, use_sample, n_per_group, sklearn_only):
        super().__init__()
        self.use_sample   = use_sample
        self.n_per_group  = n_per_group
        self.sklearn_only = sklearn_only

    def run(self):
        try:
            import random
            import pandas as pd
            from src.ml.classifier import ActivityClassifier, \
                build_sklearn_pipelines, build_all_pipelines

            random.seed(cfg.RANDOM_SEED)

            self.log_line.emit("Loading dataset...")
            df = pd.read_excel(cfg.DATASET_EXCEL)
            self.log_line.emit(
                f"Dataset: {len(df)} videos, "
                f"{df['Activity'].nunique()} activities, "
                f"{df['Distortion'].nunique()} distortions.")

            groups = {}
            for (act, dist), grp in df.groupby(["Activity", "Distortion"]):
                pool = grp["Name of Video Series"].tolist()
                groups[(act, dist)] = (
                    random.sample(pool, min(self.n_per_group, len(pool)))
                    if self.use_sample else pool
                )
            total = sum(len(v) for v in groups.values())
            self.log_line.emit(f"Processing {total} videos...")

            video_features = []
            done = 0
            for (activity, distortion), video_names in groups.items():
                for vname in video_names:
                    vpath = Path(cfg.VIDEO_DIR) / vname
                    done += 1
                    if not vpath.exists():
                        self.log_line.emit(f"  [{done}/{total}] MISSING {vname}")
                        continue

                    self.log_line.emit(
                        f"  [{done}/{total}] {activity}/{distortion}  {vname}")

                    tracks, mean_mag, mean_ang = process_video(
                        vpath, activity=activity, distortion=distortion,
                        show_preview=False)

                    track_feats = extract_all_features(
                        tracks,
                        video_name=vname, activity=activity,
                        distortion=distortion,
                        mean_flow_magnitude=mean_mag,
                        mean_flow_angle_deg=mean_ang)

                    self.log_line.emit(
                        f"    -> {len(tracks)} tracks, {len(track_feats)} valid")

                    video_fv = aggregate_video_features(track_feats)
                    if video_fv is not None:
                        video_features.append(video_fv)

            if not video_features:
                self.error.emit("No features extracted. Check video paths.")
                return

            self.log_line.emit(
                f"\nFeature matrix: {len(video_features)} videos x "
                f"{len(FEATURE_NAMES) * 3 + 1} columns")

            pipelines = (build_sklearn_pipelines() if self.sklearn_only
                         else build_all_pipelines())

            self.log_line.emit("\nTraining activity classifier...")
            clf_activity = ActivityClassifier()
            clf_activity.fit_and_evaluate(
                video_features, target_col="activity",
                run_cv=True, pipelines=pipelines,
                plot_show=False)

            self.log_line.emit("\nTraining distortion classifier...")
            clf_distortion = ActivityClassifier()
            clf_distortion.fit_and_evaluate(
                video_features, target_col="distortion",
                run_cv=True, pipelines=pipelines,
                plot_show=False)

            self.log_line.emit("\nTraining complete.")
            self.finished.emit(clf_activity, clf_distortion)

        except Exception as e:
            self.error.emit(str(e))

class VideoClassifierUI(QMainWindow):

    MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suspicious Activity Classifier")
        self.resize(1100, 820)

        self.clf_activity   = None
        self.clf_distortion = None
        self.video_path     = None
        self.worker         = None

        self.media_player = QMediaPlayer(self)

        self.build_ui()
        self.connect_signals()
        self.auto_load_model()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(self.build_classify_tab(), "Classify Video")
        tabs.addTab(self.build_train_tab(),    "Train Model")
        tabs.addTab(self.build_results_tab(),  "Model Performance")
        root.addWidget(tabs)

        self.setStatusBar(QStatusBar())
        self.media_player.setVideoOutput(self.video_widget)

    def build_classify_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        model_box = QGroupBox("Loaded Model")
        h = QHBoxLayout()
        self.lbl_model_status = QLabel("No model loaded")
        self.lbl_model_status.setStyleSheet("color: #cc0000; font-weight: bold;")
        self.btn_load_model = QPushButton("Load Model (.pkl)")
        h.addWidget(self.lbl_model_status)
        h.addStretch()
        h.addWidget(self.btn_load_model)
        model_box.setLayout(h)
        layout.addWidget(model_box)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(360)
        self.video_widget.setStyleSheet("background-color: #111;")
        layout.addWidget(self.video_widget, stretch=1)

        ctrl = QHBoxLayout()
        self.btn_upload = QPushButton("Upload Video")
        self.btn_play   = QPushButton("▶ Play")
        self.btn_play.setEnabled(False)

        self.combo_model = QComboBox()
        self.combo_model.addItem("Random Forest")
        lbl_model_pick = QLabel("Classifier:")

        self.btn_predict = QPushButton("Run Prediction")
        self.btn_predict.setEnabled(False)
        self.btn_predict.setStyleSheet(
            "QPushButton { background: #1565C0; color: white; "
            "padding: 6px 14px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:disabled { background: #aaa; }")

        ctrl.addWidget(self.btn_upload)
        ctrl.addWidget(self.btn_play)
        ctrl.addStretch()
        ctrl.addWidget(lbl_model_pick)
        ctrl.addWidget(self.combo_model)
        ctrl.addWidget(self.btn_predict)
        layout.addLayout(ctrl)

        result_box = QGroupBox("Prediction Result")
        form = QFormLayout()
        self.lbl_pred_activity = QLabel("--")
        self.lbl_pred_activity.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #1565C0;")
        self.lbl_true_activity = QLabel("--")
        self.lbl_true_activity.setStyleSheet("font-size: 14px; color: #444;")
        
        self.lbl_confidence = QLabel("--")
        form.addRow("Predicted Activity:", self.lbl_pred_activity)
        form.addRow("True Label (from filename):", self.lbl_true_activity)
        form.addRow("Confidence:", self.lbl_confidence)
        
        self.lbl_pred_distortion = QLabel("--")
        self.lbl_pred_distortion.setStyleSheet("font-size: 16px; font-weight: bold; color: #E65100;")
        form.addRow("Predicted Distortion:", self.lbl_pred_distortion)
        self.lbl_distortion_confidence = QLabel("--")
        form.addRow("Distortion Confidence:", self.lbl_distortion_confidence)


        result_box.setLayout(form)
        layout.addWidget(result_box)

        return tab

    def build_train_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        opt_box = QGroupBox("Training Options")
        opt_layout = QHBoxLayout()

        self.chk_sample = QtWidgets.QCheckBox("Sample mode")
        self.chk_sample.setChecked(True)
        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(1, 200)
        self.spin_n.setValue(cfg.N_RANDOM_VIDEOS_PER_GROUP)
        self.spin_n.setPrefix("n = ")

        self.chk_sklearn_only = QtWidgets.QCheckBox("Sklearn only (faster)")
        self.chk_sklearn_only.setChecked(False)

        self.btn_train = QPushButton("Train Model")
        self.btn_train.setStyleSheet(
            "QPushButton { background: #2E7D32; color: white; "
            "padding: 6px 18px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:disabled { background: #aaa; }")

        opt_layout.addWidget(self.chk_sample)
        opt_layout.addWidget(self.spin_n)
        opt_layout.addWidget(self.chk_sklearn_only)
        opt_layout.addStretch()
        opt_layout.addWidget(self.btn_train)
        opt_box.setLayout(opt_layout)
        layout.addWidget(opt_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        log_box = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setFont(QtGui.QFont("Monospace", 9))
        log_layout.addWidget(self.training_log)
        log_box.setLayout(log_layout)
        layout.addWidget(log_box, stretch=1)

        return tab

    def build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        acc_box = QGroupBox("Test Accuracy - All Classifiers")
        acc_layout = QVBoxLayout()
        self.lbl_accuracy_summary = QLabel("No model loaded yet.")
        self.lbl_accuracy_summary.setFont(QtGui.QFont("Monospace", 10))
        self.lbl_accuracy_summary.setWordWrap(True)
        acc_layout.addWidget(self.lbl_accuracy_summary)
        acc_box.setLayout(acc_layout)
        layout.addWidget(acc_box)

        self.fig_acc, self.ax_acc = plt.subplots(figsize=(10, 4))
        self.canvas_acc = FigureCanvas(self.fig_acc)
        self.canvas_acc.setMinimumHeight(260)
        layout.addWidget(self.canvas_acc, stretch=1)

        return tab

    def connect_signals(self):
        self.btn_load_model.clicked.connect(self.load_model_dialog)
        self.btn_upload.clicked.connect(self.upload_video)
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_predict.clicked.connect(self.run_prediction)
        self.btn_train.clicked.connect(self.start_training)

    def auto_load_model(self):
        model_path = self.MODEL_DIR / "activity.pkl"
        
        if model_path.exists():
            self.load_model_from_path(model_path)
            self.statusBar().showMessage(f"Model auto-loaded from {model_path}", 5000)

        # Also try to load distortion model
        dist_path = self.MODEL_DIR / "distortion.pkl"
        if dist_path.exists():
            try:
                with open(dist_path, "rb") as f:
                    payload = pickle.load(f)
                self.clf_distortion = payload.get("classifier")
            except Exception:
                pass


    def load_model_from_path(self, path):
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            self.clf_activity = payload.get("classifier")
            test_results      = payload.get("test_results", {})
            label_names       = payload.get("label_names", [])

            self.lbl_model_status.setText(f"Loaded: {Path(path).name}")
            self.lbl_model_status.setStyleSheet(
                "color: #2E7D32; font-weight: bold;")

            self.update_model_selector(test_results)
            self.update_results_tab(test_results, label_names)
            self.btn_predict.setEnabled(self.video_path is not None)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", str(self.MODEL_DIR),
            "Pickle files (*.pkl *.pickle)")
        if path:
            self.load_model_from_path(path)

    def update_model_selector(self, test_results):
        self.combo_model.clear()
        for name in test_results:
            self.combo_model.addItem(name)
        if not test_results:
            self.combo_model.addItem("Random Forest")

    def update_results_tab(self, test_results, label_names):
        if not test_results:
            return

        lines = [f"{'Classifier':<30}  {'Test Accuracy':>14}"]
        lines.append("-" * 46)
        for name, acc in sorted(test_results.items(),
                                 key=lambda x: x[1], reverse=True):
            lines.append(f"  {name:<28}  {acc:.4f}  ({acc*100:.1f}%)")
        self.lbl_accuracy_summary.setText("\n".join(lines))

        self.ax_acc.clear()
        names  = list(test_results.keys())
        accs   = [test_results[n] for n in names]
        colors = ["#1565C0" if "Custom" not in n else "#E65100" for n in names]
        bars   = self.ax_acc.bar(names, accs, color=colors)
        self.ax_acc.set_ylim(0, 1.1)
        self.ax_acc.set_ylabel("Test Accuracy")
        self.ax_acc.set_title(
            "Classifier Comparison  (blue = sklearn, orange = custom)")
        self.ax_acc.tick_params(axis="x", rotation=25)
        for bar, acc in zip(bars, accs):
            self.ax_acc.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=8)
        self.fig_acc.tight_layout()
        self.canvas_acc.draw()

    def upload_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        self.video_path = path
        self.media_player.setMedia(
            QMediaContent(QUrl.fromLocalFile(path)))
        self.media_player.setPosition(0)
        self.btn_play.setEnabled(True)
        self.btn_play.setText("▶ Play")
        self.btn_predict.setEnabled(self.clf_activity is not None)
        self.statusBar().showMessage(f"Video loaded: {Path(path).name}")

        true_label = self.parse_true_label(Path(path).name)
        self.lbl_true_activity.setText(true_label or "unknown (not in filename)")

    def toggle_playback(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.btn_play.setText("▶ Play")
        else:
            self.media_player.play()
            self.btn_play.setText("⏸ Pause")

    def parse_true_label(self, filename):
        label_map = {
            "LPP": "Leaving Package in a Public Place (LPP)",
            "PO":  "Passing Out (PO)",
            "PW":  "Prowl (PW)",
            "PPP": "Person Pushing Person (PPP)",
            "RK":  "Robbery with Knife (RK)",
            "FG":  "Fighting in Group (FG)",
            "PR":  "Person Running (PR)",
            "WL":  "Walking (WL)",
        }
        stem = Path(filename).stem.upper()
        for code, full in label_map.items():
            if code in stem:
                return full
        return None

    def predict_distortion(self, video_fv, model_name):
        X = np.array([video_fv[f] for f in FEATURE_NAMES]).reshape(1, -1)
        if self.clf_distortion is None:
            return "No distortion model", None
        try:
            pipe = self.clf_distortion.pipelines[model_name]
            pred_encoded = pipe.predict(X)[0]
            label = self.clf_distortion.encoder.inverse_transform(
                [pred_encoded])[0]

            confidence = None
            clf_step = pipe.named_steps.get("clf")
            if clf_step and hasattr(clf_step, "predict_proba"):
                X_scaled = pipe.named_steps["scaler"].transform(X)
                proba = clf_step.predict_proba(X_scaled)[0]
                confidence = float(proba[pred_encoded])

            return label, confidence
        except Exception as e:
            return f"Error: {e}", None
    
    def run_prediction(self):
        if not self.clf_activity:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return
        if not self.video_path:
            QMessageBox.warning(self, "No Video", "Please upload a video first.")
            return

        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.btn_play.setText("▶ Play")

        self.statusBar().showMessage(
            "Processing video - this may take a few seconds...")
        self.btn_predict.setEnabled(False)

        try:
            tracks, mean_mag, mean_ang = process_video(
                Path(self.video_path), show_preview=False)

            track_feats = extract_all_features(
                tracks,
                video_name=Path(self.video_path).name,
                activity="", distortion="",
                mean_flow_magnitude=mean_mag,
                mean_flow_angle_deg=mean_ang)

            if not track_feats:
                QMessageBox.warning(self, "No Motion",
                                    "No motion tracks detected in the video.\n"
                                    "The video may be too short or too distorted.")
                return

            video_fv = aggregate_video_features(track_feats)
            model_name = self.combo_model.currentText()

            # Predict activity
            pred_label, confidence = self.predict(video_fv, model_name)
            self.lbl_pred_activity.setText(str(pred_label))
            self.lbl_confidence.setText(
                f"{confidence:.1%}" if confidence is not None else "N/A")

            # Predict distortion
            dist_label, dist_conf = self.predict_distortion(video_fv, model_name)
            self.lbl_pred_distortion.setText(str(dist_label))
            if hasattr(self, 'lbl_distortion_confidence'):
                self.lbl_distortion_confidence.setText(
                    f"{dist_conf:.1%}" if dist_conf is not None else "N/A")

            self.statusBar().showMessage("Prediction complete!", 5000)


        except Exception as e:
            QMessageBox.critical(self, "Prediction Failed", str(e))
            self.statusBar().showMessage("Prediction failed.")
        finally:
            self.btn_predict.setEnabled(True)

    def predict(self, video_fv, model_name):
        X = np.array([video_fv[f] for f in FEATURE_NAMES]).reshape(1, -1)
        try:
            pipe = self.clf_activity.pipelines[model_name]
            pred_encoded = pipe.predict(X)[0]
            label = self.clf_activity.encoder.inverse_transform(
                [pred_encoded])[0]

            confidence = None
            clf_step   = pipe.named_steps.get("clf")
            if clf_step and hasattr(clf_step, "predict_proba"):
                X_scaled = pipe.named_steps["scaler"].transform(X)
                proba    = clf_step.predict_proba(X_scaled)[0]
                confidence = float(proba[pred_encoded])

            return label, confidence
        except Exception as e:
            return f"Error: {e}", None

    def start_training(self):
        model_path = self.MODEL_DIR / "activity.pkl"
        if model_path.exists():
            ans = QMessageBox.question(
                self, "Override Model?",
                f"A trained model already exists at:\n{model_path}\n\n"
                "Do you want to retrain and override it?",
                QMessageBox.Yes | QMessageBox.No)
            if ans != QMessageBox.Yes:
                return

        self.training_log.clear()
        self.btn_train.setEnabled(False)
        self.progress_bar.setVisible(True)

        self.worker = TrainingWorker(
            use_sample   = self.chk_sample.isChecked(),
            n_per_group  = self.spin_n.value(),
            sklearn_only = self.chk_sklearn_only.isChecked(),
        )
        self.worker.log_line.connect(self.append_log)
        self.worker.finished.connect(self.on_training_done)
        self.worker.error.connect(self.on_training_error)
        self.worker.start()

    def append_log(self, line):
        self.training_log.append(line)
        sb = self.training_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_training_done(self, clf_activity, clf_distortion):
        self.clf_activity   = clf_activity
        self.clf_distortion = clf_distortion

        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        clf_activity.save(self.MODEL_DIR / "activity.pkl",
                          target_col="activity")
        clf_distortion.save(self.MODEL_DIR / "distortion.pkl",
                            target_col="distortion")

        self.lbl_model_status.setText("Loaded: activity.pkl (just trained)")
        self.lbl_model_status.setStyleSheet("color: #2E7D32; font-weight: bold;")
        self.update_model_selector(clf_activity.test_results)
        self.update_results_tab(clf_activity.test_results,
                                clf_activity.label_names)
        self.btn_predict.setEnabled(self.video_path is not None)
        self.btn_train.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Training complete. Model saved.", 6000)
        self.append_log("\nModel saved to models/activity.pkl")

    def on_training_error(self, msg):
        self.btn_train.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Training Error", msg)
        self.statusBar().showMessage("Training failed.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = VideoClassifierUI()
    win.show()
    sys.exit(app.exec_())