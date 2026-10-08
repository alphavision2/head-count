import os
import sys

_DEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps")
if os.path.isdir(_DEPS_DIR) and _DEPS_DIR not in sys.path:
    sys.path.insert(0, _DEPS_DIR)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
import csv

from cascade_utils import open_camera

from face_recognition_module import (
    load_students,
    register_student,
    delete_student,
    recognize_faces_in_image,
    recognize_faces_from_frame
)


class ClassroomHeadCountApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classroom Head Count System - Face Recognition")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        self.cap = None
        self.webcam_running = False
        self.current_frame = None
        self.detected_present_ids = set()

        self._setup_styles()
        self._build_ui()
        self.refresh_student_list()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'))
        style.configure('Subtitle.TLabel', font=('Segoe UI', 11))
        style.configure('Header.TLabelframe.Label', font=('Segoe UI', 11, 'bold'))
        style.configure('Big.TButton', font=('Segoe UI', 11, 'bold'), padding=8)
        style.configure('Count.TLabel', font=('Segoe UI', 28, 'bold'), foreground='#2563eb')

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.tab_home = ttk.Frame(notebook)
        self.tab_register = ttk.Frame(notebook)
        self.tab_attendance = ttk.Frame(notebook)

        notebook.add(self.tab_home, text="  Home  ")
        notebook.add(self.tab_register, text="  Register Student  ")
        notebook.add(self.tab_attendance, text="  Take Attendance  ")

        self._build_home_tab()
        self._build_register_tab()
        self._build_attendance_tab()

    def _build_home_tab(self):
        frame = self.tab_home

        title = ttk.Label(frame, text="Classroom Head Count System", style='Title.TLabel')
        title.pack(pady=(30, 8))

        subtitle = ttk.Label(frame, text="Face Recognition Based Student Attendance", style='Subtitle.TLabel')
        subtitle.pack(pady=(0, 30))

        stats_frame = ttk.Frame(frame)
        stats_frame.pack(pady=20)

        self.stat_total = self._make_stat_card(stats_frame, "Total Registered", "0", 0)
        self.stat_today = self._make_stat_card(stats_frame, "Present Today", "0", 1)
        self.stat_detected = self._make_stat_card(stats_frame, "Faces Detected", "0", 2)

        self._refresh_home_stats()

        quick_frame = tk.Frame(frame, bg='#ecfdf5', highlightbackground='#a7f3d0', highlightthickness=1)
        quick_frame.pack(padx=40, pady=(0, 10), fill='x')
        ttk.Label(quick_frame, text="Quick: Just count faces in front of webcam (no registration needed)",
                  background='#ecfdf5', foreground='#065f46', font=('Segoe UI', 10, 'bold')
                  ).pack(side='left', padx=15, pady=12)
        self.btn_quick_count = ttk.Button(quick_frame, text="Start Quick Count", style='Big.TButton',
                                          command=self.toggle_quick_count)
        self.btn_quick_count.pack(side='right', padx=15, pady=10)
        self._quick_count_running = False

        info = ttk.LabelFrame(frame, text=" How to Use ", style='Header.TLabelframe')
        info.pack(padx=40, pady=20, fill='x')

        steps = [
            "1. Go to 'Register Student' tab to add students by uploading their face photos.",
            "2. Go to 'Take Attendance' tab to capture the classroom - either upload a group photo or use webcam.",
            "3. The system will detect and recognize all registered students and count them as present.",
            "4. You can export the attendance record as CSV file."
        ]
        for s in steps:
            ttk.Label(info, text=s, font=('Segoe UI', 10)).pack(anchor='w', padx=15, pady=4)

    def _make_stat_card(self, parent, label_text, value_text, col):
        card = tk.Frame(parent, bg='#f0f4ff', highlightbackground='#dbeafe', highlightthickness=1)
        card.grid(row=0, column=col, padx=20, ipadx=30, ipady=20)

        val_lbl = tk.Label(card, text=value_text, font=('Segoe UI', 28, 'bold'), fg='#2563eb', bg='#f0f4ff')
        val_lbl.pack(pady=(5, 0))

        lbl = tk.Label(card, text=label_text, font=('Segoe UI', 11), bg='#f0f4ff', fg='#374151')
        lbl.pack(pady=(0, 5))

        return val_lbl

    def _refresh_home_stats(self):
        students = load_students()
        self.stat_total.config(text=str(len(students)))

    def _build_register_tab(self):
        frame = self.tab_register

        form_frame = ttk.LabelFrame(frame, text=" Register New Student ", style='Header.TLabelframe')
        form_frame.pack(fill='x', padx=15, pady=10)

        form_inner = ttk.Frame(form_frame)
        form_inner.pack(padx=15, pady=15, fill='x')

        ttk.Label(form_inner, text="Student ID:", font=('Segoe UI', 10)).grid(row=0, column=0, sticky='w', pady=6)
        self.entry_id = ttk.Entry(form_inner, font=('Segoe UI', 10), width=30)
        self.entry_id.grid(row=0, column=1, sticky='w', pady=6, padx=10)

        ttk.Label(form_inner, text="Student Name:", font=('Segoe UI', 10)).grid(row=1, column=0, sticky='w', pady=6)
        self.entry_name = ttk.Entry(form_inner, font=('Segoe UI', 10), width=30)
        self.entry_name.grid(row=1, column=1, sticky='w', pady=6, padx=10)

        ttk.Label(form_inner, text="Face Photo:", font=('Segoe UI', 10)).grid(row=2, column=0, sticky='w', pady=6)
        self.entry_image = ttk.Entry(form_inner, font=('Segoe UI', 10), width=30)
        self.entry_image.grid(row=2, column=1, sticky='w', pady=6, padx=10)
        ttk.Button(form_inner, text="Browse...", command=self.browse_image).grid(row=2, column=2, padx=5)

        self.reg_preview_canvas = tk.Canvas(form_inner, width=180, height=180, bg='#e5e7eb', highlightthickness=1, highlightbackground='#9ca3af')
        self.reg_preview_canvas.grid(row=0, column=3, rowspan=3, padx=20)

        btn_frame = ttk.Frame(form_inner)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=(15, 0), sticky='w')
        ttk.Button(btn_frame, text="Register Student", style='Big.TButton', command=self.do_register).pack(side='left')
        ttk.Button(btn_frame, text="Clear Form", command=self.clear_reg_form).pack(side='left', padx=10)

        list_frame = ttk.LabelFrame(frame, text=" Registered Students ", style='Header.TLabelframe')
        list_frame.pack(fill='both', expand=True, padx=15, pady=10)

        cols = ('id', 'name')
        self.student_tree = ttk.Treeview(list_frame, columns=cols, show='headings', height=12)
        self.student_tree.heading('id', text='Student ID')
        self.student_tree.heading('name', text='Student Name')
        self.student_tree.column('id', width=150)
        self.student_tree.column('name', width=300)
        self.student_tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)

        scroll = ttk.Scrollbar(list_frame, orient='vertical', command=self.student_tree.yview)
        self.student_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side='left', fill='y', pady=10)

        action_frame = ttk.Frame(list_frame)
        action_frame.pack(side='left', padx=15, pady=10, fill='y')
        ttk.Button(action_frame, text="Delete Selected", command=self.delete_selected_student).pack(fill='x', pady=5)
        ttk.Button(action_frame, text="Refresh List", command=self.refresh_student_list).pack(fill='x', pady=5)

    def _build_attendance_tab(self):
        frame = self.tab_attendance

        top_bar = ttk.Frame(frame)
        top_bar.pack(fill='x', padx=15, pady=(10, 5))

        ttk.Button(top_bar, text="Upload Group Photo", style='Big.TButton', command=self.upload_group_photo).pack(side='left')
        ttk.Button(top_bar, text="Start Webcam", command=self.start_webcam).pack(side='left', padx=10)
        ttk.Button(top_bar, text="Stop Webcam", command=self.stop_webcam).pack(side='left')
        ttk.Button(top_bar, text="Capture & Count", command=self.capture_and_count).pack(side='left', padx=10)
        ttk.Button(top_bar, text="Export CSV", command=self.export_csv).pack(side='right')

        main = ttk.Frame(frame)
        main.pack(fill='both', expand=True, padx=15, pady=5)

        vid_frame = ttk.LabelFrame(main, text=" Camera / Image View ", style='Header.TLabelframe')
        vid_frame.pack(side='left', fill='both', expand=True)

        self.video_canvas = tk.Canvas(vid_frame, bg='#1f2937')
        self.video_canvas.pack(fill='both', expand=True, padx=8, pady=8)

        result_frame = ttk.LabelFrame(main, text=" Attendance Result ", style='Header.TLabelframe')
        result_frame.pack(side='left', fill='both', expand=False, padx=(10, 0))
        result_frame.configure(width=360)

        counts = ttk.Frame(result_frame)
        counts.pack(fill='x', padx=12, pady=10)

        self.lbl_count_present = tk.Label(counts, text="0", font=('Segoe UI', 22, 'bold'), fg='#059669')
        self.lbl_count_present.grid(row=0, column=0, padx=12)
        tk.Label(counts, text="Present\n(Recognized)", font=('Segoe UI', 9), fg='#374151',
                 justify='center').grid(row=1, column=0, padx=12)

        tk.Label(counts, text="/", font=('Segoe UI', 22, 'bold'), fg='#9ca3af').grid(row=0, column=1)
        tk.Label(counts, text="", font=('Segoe UI', 10)).grid(row=1, column=1)

        self.lbl_count_total = tk.Label(counts, text="0", font=('Segoe UI', 22, 'bold'), fg='#2563eb')
        self.lbl_count_total.grid(row=0, column=2, padx=12)
        tk.Label(counts, text="Registered", font=('Segoe UI', 9), fg='#374151').grid(row=1, column=2, padx=12)

        totals2 = tk.Frame(result_frame, bg='#fffbeb', highlightbackground='#fde68a', highlightthickness=1)
        totals2.pack(fill='x', padx=10, pady=(5, 5))
        self.lbl_count_all = tk.Label(totals2, text="0", font=('Segoe UI', 18, 'bold'), fg='#b45309', bg='#fffbeb')
        self.lbl_count_all.pack(side='left', padx=(15, 8), pady=6)
        tk.Label(totals2, text="Total People in front of Camera\n(registered + unknown)",
                 font=('Segoe UI', 9), bg='#fffbeb', fg='#78350f', justify='left'
                 ).pack(side='left', pady=6)

        self.lbl_count_unknown = tk.Label(result_frame, text="Unknown faces: 0", font=('Segoe UI', 10), fg='#b45309')
        self.lbl_count_unknown.pack(pady=(0, 5))

        list_cols = ('id', 'name')
        self.present_tree = ttk.Treeview(result_frame, columns=list_cols, show='headings', height=16)
        self.present_tree.heading('id', text='ID')
        self.present_tree.heading('name', text='Name')
        self.present_tree.column('id', width=90)
        self.present_tree.column('name', width=230)
        self.present_tree.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        self._refresh_attendance_totals()

    def _refresh_attendance_totals(self):
        students = load_students()
        self.lbl_count_total.config(text=str(len(students)))

    # ---------------- REGISTER TAB ACTIONS ----------------
    def browse_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if path:
            self.entry_image.delete(0, tk.END)
            self.entry_image.insert(0, path)
            self._show_image_preview(path, self.reg_preview_canvas, 180, 180)

    def _show_image_preview(self, path, canvas, w, h):
        try:
            img = Image.open(path)
            img.thumbnail((w, h), Image.LANCZOS)
            canvas._photo = ImageTk.PhotoImage(img)
            canvas.delete('all')
            cw, chl = canvas.winfo_width(), canvas.winfo_height()
            cw = cw if cw > 1 else w
            chl = chl if chl > 1 else h
            x = (cw - img.width) // 2
            y = (chl - img.height) // 2
            canvas.create_image(x if x > 0 else 0, y if y > 0 else 0, anchor='nw', image=canvas._photo)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def do_register(self):
        sid = self.entry_id.get().strip()
        name = self.entry_name.get().strip()
        img = self.entry_image.get().strip()

        if not sid or not name or not img:
            messagebox.showwarning("Missing Info", "Please fill Student ID, Name and select a face photo.")
            return

        if not os.path.exists(img):
            messagebox.showerror("Error", "Image file not found.")
            return

        success, msg = register_student(sid, name, img)
        if success:
            messagebox.showinfo("Success", msg)
            self.clear_reg_form()
            self.refresh_student_list()
            self._refresh_home_stats()
            self._refresh_attendance_totals()
        else:
            messagebox.showerror("Registration Failed", msg)

    def clear_reg_form(self):
        self.entry_id.delete(0, tk.END)
        self.entry_name.delete(0, tk.END)
        self.entry_image.delete(0, tk.END)
        self.reg_preview_canvas.delete('all')

    def refresh_student_list(self):
        for i in self.student_tree.get_children():
            self.student_tree.delete(i)
        students = load_students()
        for sid in sorted(students.keys()):
            self.student_tree.insert('', 'end', values=(sid, students[sid]['name']))

    def delete_selected_student(self):
        sel = self.student_tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Select a student from the list to delete.")
            return
        item = self.student_tree.item(sel[0])
        sid = item['values'][0]
        if messagebox.askyesno("Confirm Delete", f"Delete student ID {sid}? This cannot be undone."):
            ok, msg = delete_student(sid)
            if ok:
                messagebox.showinfo("Deleted", msg)
                self.refresh_student_list()
                self._refresh_home_stats()
                self._refresh_attendance_totals()
            else:
                messagebox.showerror("Error", msg)

    # ---------------- ATTENDANCE TAB ACTIONS ----------------
    def upload_group_photo(self):
        self.stop_webcam()
        self._stop_quick_count()
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return

        try:
            frame = cv2.imread(path)
            if frame is None:
                messagebox.showerror("Error", "Could not read the image.")
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb
            self._display_frame(frame_rgb)

            result, err = recognize_faces_in_image(path)
            if err:
                messagebox.showwarning("Notice", err)
                return
            self._show_attendance_result(result)
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {e}")

    def start_webcam(self):
        if self.webcam_running:
            return
        self._stop_quick_count()
        try:
            self.cap = open_camera(0, width=960, height=720)
            if self.cap is None:
                raise RuntimeError(
                    "Could not open any webcam.\n\n"
                    "Close other apps that may use your camera (Zoom, Teams, Chrome, etc.),\n"
                    "or try plugging the camera into a different USB port."
                )
        except Exception as e:
            messagebox.showerror("Webcam Error", str(e))
            self.cap = None
            return

        self.webcam_running = True
        self.detected_present_ids = set()
        self._clear_attendance_list()
        self._update_webcam_loop()

    def stop_webcam(self):
        self.webcam_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _update_webcam_loop(self):
        if not self.webcam_running or not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb.copy()

            try:
                face_locations, face_matched_ids, present_ids = recognize_faces_from_frame(frame_rgb)
                for pid in present_ids:
                    self.detected_present_ids.add(pid)

                students = load_students()
                known_count = 0
                unknown_count = 0
                for idx, (top, right, bottom, left) in enumerate(face_locations):
                    matched_id = face_matched_ids[idx] if idx < len(face_matched_ids) else None
                    if matched_id and matched_id in students:
                        color = (5, 150, 105)
                        label = students[matched_id]['name']
                        known_count += 1
                    else:
                        color = (180, 83, 9)
                        label = "Unknown"
                        unknown_count += 1
                    cv2.rectangle(frame_rgb, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame_rgb, label, (left, max(top - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                total_faces = len(face_locations)
                banner_h = 90
                hdr = np.zeros((banner_h, frame_rgb.shape[1], 3), dtype=np.uint8)
                hdr[:] = (15, 23, 42)
                cv2.putText(hdr, f"TOTAL PEOPLE = {total_faces}", (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (251, 191, 36), 3)
                cv2.putText(hdr,
                            f"Recognized: {known_count}   |   Unknown: {unknown_count}",
                            (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (209, 213, 219), 2)
                disp = np.concatenate((hdr, frame_rgb), axis=0)

                self._display_frame(disp)
                self.lbl_count_all.config(text=str(total_faces))
                self.lbl_count_unknown.config(text=f"Unknown faces: {unknown_count}")
            except Exception:
                self._display_frame(frame_rgb)

            self._update_live_present_display()

        self.root.after(40, self._update_webcam_loop)

    def _display_frame(self, rgb_frame):
        canvas = self.video_canvas
        cw = max(canvas.winfo_width(), 100)
        ch = max(canvas.winfo_height(), 100)

        img = Image.fromarray(rgb_frame)
        img.thumbnail((cw, ch), Image.LANCZOS)

        canvas._photo = ImageTk.PhotoImage(img)
        canvas.delete('all')
        x = (cw - img.width) // 2
        y = (ch - img.height) // 2
        canvas.create_image(x if x > 0 else 0, y if y > 0 else 0, anchor='nw', image=canvas._photo)

    def capture_and_count(self):
        if self.current_frame is None:
            messagebox.showinfo("No Image", "Please start the webcam first or upload a group photo.")
            return

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tmp.close()
        bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp.name, bgr)

        result, err = recognize_faces_in_image(tmp.name)
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

        if err:
            messagebox.showwarning("Notice", err)
            return

        for d in result['present']:
            self.detected_present_ids.add(d['id'])

        students = load_students()
        present_list = [students[sid] for sid in self.detected_present_ids if sid in students]
        merged = {
            'present': present_list,
            'present_count': len(present_list),
            'total_registered': len(students),
            'unknown_faces': result['unknown_faces']
        }
        self._show_attendance_result(merged)

    def _update_live_present_display(self):
        students = load_students()
        present = [students[sid] for sid in sorted(self.detected_present_ids) if sid in students]
        self.lbl_count_present.config(text=str(len(present)))
        self.lbl_count_total.config(text=str(len(students)))

        for i in self.present_tree.get_children():
            self.present_tree.delete(i)
        for s in present:
            self.present_tree.insert('', 'end', values=(s['id'], s['name']))

    def _clear_attendance_list(self):
        for i in self.present_tree.get_children():
            self.present_tree.delete(i)
        self.lbl_count_present.config(text="0")
        self.lbl_count_unknown.config(text="Unknown faces: 0")
        self.lbl_count_all.config(text="0")

    def _show_attendance_result(self, result):
        for i in self.present_tree.get_children():
            self.present_tree.delete(i)
        for s in result['present']:
            self.present_tree.insert('', 'end', values=(s['id'], s['name']))

        self.lbl_count_present.config(text=str(result['present_count']))
        self.lbl_count_total.config(text=str(result['total_registered']))
        unknown = result.get('unknown_faces', 0)
        total_detected = result.get('total_faces_detected', 0)
        self.lbl_count_unknown.config(text=f"Unknown faces: {unknown}")
        self.lbl_count_all.config(text=str(total_detected))

        if 'stat_detected' in self.__dict__:
            self.stat_detected.config(text=str(total_detected))
        self.stat_today.config(text=str(result['present_count']))

    def export_csv(self):
        items = self.present_tree.get_children()
        if not items:
            messagebox.showinfo("No Data", "No attendance records to export. Take attendance first.")
            return

        default_name = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=default_name,
                                            filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        students = load_students()
        present_ids = set()
        for it in items:
            vals = self.present_tree.item(it, 'values')
            present_ids.add(vals[0])

        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Student ID', 'Name', 'Status', 'Timestamp'])
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for sid in sorted(students.keys()):
                status = "Present" if sid in present_ids else "Absent"
                w.writerow([sid, students[sid]['name'], status, ts])

        messagebox.showinfo("Exported", f"Attendance exported to:\n{path}")

    def on_close(self):
        self.stop_webcam()
        self._stop_quick_count()
        self.root.destroy()

    # ---------- Quick Count (Haar face counter, no registration) ----------
    def toggle_quick_count(self):
        if self._quick_count_running:
            self._stop_quick_count()
        else:
            self._start_quick_count()

    def _start_quick_count(self):
        self.stop_webcam()
        from cascade_utils import build_haar_cascade
        self._quick_cascade, used_path = build_haar_cascade()
        if self._quick_cascade is None or self._quick_cascade.empty():
            messagebox.showerror(
                "Face Detector Error",
                "Failed to load the face detector (Haar cascade).\n\n"
                f"Checked location: {used_path}\n"
                "Make sure you're connected to the internet on first run so the\n"
                "cascade file can be downloaded automatically into cascades/ folder."
            )
            return

        try:
            self._quick_cap = open_camera(0, width=960, height=720)
            if self._quick_cap is None:
                raise RuntimeError(
                    "Could not open any webcam.\n\n"
                    "Close other apps that may use your camera (Zoom, Teams, Chrome, etc.),\n"
                    "or try plugging the camera into a different USB port."
                )
        except Exception as e:
            messagebox.showerror("Webcam Error", str(e))
            self._quick_cap = None
            return

        self._quick_count_running = True
        self.btn_quick_count.config(text="Stop Quick Count")
        self._quick_last_counts = []
        self._quick_smoothed = 0
        self._quick_count_loop()

    def _stop_quick_count(self):
        if not getattr(self, '_quick_count_running', False):
            return
        self._quick_count_running = False
        self.btn_quick_count.config(text="Start Quick Count")
        cap = getattr(self, '_quick_cap', None)
        if cap:
            try:
                cap.release()
            except Exception:
                pass
            self._quick_cap = None

    def _quick_count_loop(self):
        if not self._quick_count_running:
            return
        cap = getattr(self, '_quick_cap', None)
        cascade = getattr(self, '_quick_cascade', None)
        if not cap or not cascade:
            self._stop_quick_count()
            return
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb.copy()
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                             minSize=(90, 90), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 200, 90), 2)

            cur = len(faces)
            self._quick_last_counts.append(cur)
            if len(self._quick_last_counts) > 6:
                self._quick_last_counts.pop(0)
            avg = int(round(sum(self._quick_last_counts) / len(self._quick_last_counts)))
            self._quick_smoothed = int(round(0.65 * self._quick_smoothed + 0.35 * avg))

            disp = frame_rgb.copy()
            hdr = np.zeros((100, disp.shape[1], 3), dtype=np.uint8)
            hdr[:] = (15, 23, 42)
            disp = np.concatenate((hdr, disp), axis=0)
            cv2.putText(disp, f"People in front of camera: COUNT = {self._quick_smoothed}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 140), 3)

            self._display_frame(disp)
            self.stat_detected.config(text=str(self._quick_smoothed))
            self.stat_today.config(text=str(self._quick_smoothed))

        self.root.after(35, self._quick_count_loop)


def main():
    root = tk.Tk()
    app = ClassroomHeadCountApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
