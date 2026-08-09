"""
Fallback GUI - Pure Python HUD without Electron
למקרה שאין Electron - HUD ב-PyQt5/Tkinter בסגנון Iron Man
ניתן להריץ כ-exe עצמאי עם PyInstaller
"""
import sys
import os
import threading
import time
import webbrowser
from pathlib import Path

# נסה PyQt6 / PyQt5 / Tkinter לפי זמינות
GUI_BACKEND = None
try:
    from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint
    from PyQt6.QtGui import QPainter, QColor, QPen, QRadialGradient
    GUI_BACKEND = "pyqt6"
except:
    try:
        from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint
        from PyQt5.QtGui import QPainter, QColor, QPen, QRadialGradient
        GUI_BACKEND = "pyqt5"
    except:
        try:
            import tkinter as tk
            GUI_BACKEND = "tkinter"
        except:
            GUI_BACKEND = None

print(f"[GUI Fallback] Backend: {GUI_BACKEND}")

if GUI_BACKEND in ("pyqt6", "pyqt5"):
    class ReactorWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.setFixedSize(120, 120)
            self.angle = 0
            self.mode = "idle"  # idle, listening, speaking
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_anim)
            self.timer.start(30)

        def update_anim(self):
            self.angle += 0.05
            if self.angle > 6.28:
                self.angle = 0
            self.update()

        def set_mode(self, mode):
            self.mode = mode
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            center = QPoint(60, 60)
            
            # Glow
            if self.mode == "listening":
                grad = QRadialGradient(center, 60)
                grad.setColorAt(0, QColor(255, 204, 0, 80))
                grad.setColorAt(1, QColor(0, 0, 0, 0))
            elif self.mode == "speaking":
                grad = QRadialGradient(center, 60)
                grad.setColorAt(0, QColor(0, 210, 255, 100))
                grad.setColorAt(1, QColor(0, 0, 0, 0))
            else:
                grad = QRadialGradient(center, 60)
                grad.setColorAt(0, QColor(0, 210, 255, 60))
                grad.setColorAt(1, QColor(0, 0, 0, 0))
            
            painter.setBrush(grad)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(0, 0, 120, 120)

            # Outer ring
            pen = QPen(QColor(0, 210, 255, 180), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(10, 10, 100, 100)

            # Inner triangle - rotating
            painter.save()
            painter.translate(60, 60)
            painter.rotate(self.angle * 30 / 3.14)
            
            if self.mode == "listening":
                painter.setBrush(QColor(255, 204, 0))
            else:
                painter.setBrush(QColor(0, 210, 255))
            painter.setPen(Qt.PenStyle.NoPen)
            
            from math import cos, sin, pi
            points = []
            for i in range(3):
                a = pi*2/3 * i - pi/2
                points.append(QPoint(int(cos(a)*20), int(sin(a)*20)))
            # Draw triangle manually
            from PyQt6.QtGui import QPolygon
            if GUI_BACKEND == "pyqt6":
                from PyQt6.QtCore import QPointF
                # simplified - draw circle instead for compat
                painter.drawEllipse(-12, -12, 24, 24)
            else:
                painter.drawEllipse(-12, -12, 24, 24)
            painter.restore()

    class AdielWindow(QMainWindow):
        def __init__(self, backend_url="http://localhost:8765"):
            super().__init__()
            self.backend_url = backend_url
            self.setWindowTitle("Adiel Junior - אדיאל ג'וניור")
            self.setFixedSize(480, 640)
            
            # Frameless + transparent + always on top
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            
            central = QWidget()
            central.setObjectName("central")
            central.setStyleSheet("""
                #central {
                    background: rgba(8, 14, 28, 0.88);
                    border: 1px solid rgba(0, 210, 255, 0.3);
                    border-radius: 20px;
                }
                QLabel { color: #e3f2ff; }
            """)
            self.setCentralWidget(central)

            layout = QVBoxLayout(central)
            layout.setContentsMargins(16, 16, 16, 16)

            # Header
            header = QHBoxLayout()
            title = QLabel("ADIEL JUNIOR")
            title.setStyleSheet("font-weight: bold; letter-spacing: 2px; font-size: 12px;")
            header.addWidget(title)
            header.addStretch()

            btn_min = QPushButton("—")
            btn_close = QPushButton("✕")
            for btn in (btn_min, btn_close):
                btn.setFixedSize(28, 28)
                btn.setStyleSheet("background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: #8aa0b8;")
            btn_min.clicked.connect(self.showMinimized)
            btn_close.clicked.connect(self.close)
            header.addWidget(btn_min)
            header.addWidget(btn_close)
            layout.addLayout(header)

            # Reactor
            self.reactor = ReactorWidget()
            layout.addWidget(self.reactor, alignment=Qt.AlignmentFlag.AlignCenter)

            self.status = QLabel("מאזינה למילת הפעלה 'אדיאל ג'וניור'...")
            self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.status.setStyleSheet("font-size: 14px; padding: 8px;")
            layout.addWidget(self.status)

            self.chat = QTextEdit()
            self.chat.setReadOnly(True)
            self.chat.setStyleSheet("""
                background: rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 12px;
                color: #e3f2ff;
                padding: 8px;
            """)
            self.chat.setPlaceholderText("שיחה עם אדיאל...")
            layout.addWidget(self.chat, stretch=1)

            # Input placeholder
            self.chat.append("שלום בוס! זה ה-HUD החלופי ב-PyQt.\nאם יש לך Electron, השתמש בו לחוויה המלאה.\nאמור 'אדיאל ג'וניור' כדי להתחיל.\n")

            # Drag support
            self.drag_pos = None

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft() if hasattr(event, 'globalPosition') else event.globalPos() - self.frameGeometry().topLeft()

        def mouseMoveEvent(self, event):
            if self.drag_pos and event.buttons() == Qt.MouseButton.LeftButton:
                new_pos = (event.globalPosition().toPoint() - self.drag_pos) if hasattr(event, 'globalPosition') else (event.globalPos() - self.drag_pos)
                self.move(new_pos)

        def mouseReleaseEvent(self, event):
            self.drag_pos = None

        def log(self, text):
            self.chat.append(text)

        def set_listening(self, listening=True):
            if listening:
                self.status.setText("🎤 מקשיבה...")
                self.reactor.set_mode("listening")
            else:
                self.status.setText("מאזינה למילת הפעלה...")
                self.reactor.set_mode("idle")

    def run_qt_gui():
        app = QApplication(sys.argv)
        win = AdielWindow()
        win.show()
        sys.exit(app.exec() if hasattr(app, 'exec') else app.exec_())

elif GUI_BACKEND == "tkinter":
    import tkinter as tk

    def run_tk_gui():
        root = tk.Tk()
        root.title("Adiel Junior")
        root.geometry("480x640")
        root.attributes('-topmost', True)
        root.configure(bg='#080e1c')
        # Frameless not trivial in Tk, we keep normal window
        # root.overrideredirect(True)  # frameless
        
        label = tk.Label(root, text="ADIEL JUNIOR\nאדיאל ג'וניור", fg="#00d2ff", bg="#080e1c", font=("Heebo", 14))
        label.pack(pady=20)
        
        status = tk.Label(root, text="מאזינה למילת הפעלה 'אדיאל ג'וניור'...", fg="#e3f2ff", bg="#080e1c", wraplength=400)
        status.pack(pady=10)
        
        chat = tk.Text(root, height=20, bg="#0a1428", fg="#e3f2ff", insertbackground="#00d2ff")
        chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        chat.insert(tk.END, "שלום בוס! HUD חלופי ב-Tkinter.\nלהפעלה מלאה, התקן Electron:\ncd frontend && npm install && npm start\n\n")
        
        def on_wake():
            status.config(text="כן בוס? אני כאן!")
            chat.insert(tk.END, "\n[WAKE] אדיאל ג'וניור התעוררה!\n")
            chat.see(tk.END)
        
        btn = tk.Button(root, text="🎤 אדיאל ג'וניור", command=on_wake, bg="#00d2ff", fg="black", font=("Heebo", 12, "bold"))
        btn.pack(pady=10)
        
        root.mainloop()

    def run_qt_gui():
        run_tk_gui()

else:
    def run_qt_gui():
        print("No GUI backend available! Install PyQt6: pip install PyQt6")
        input("Press Enter to exit...")

if __name__ == "__main__":
    print(f"Starting Adiel Junior Fallback HUD with {GUI_BACKEND}")
    run_qt_gui()
