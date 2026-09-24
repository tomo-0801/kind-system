import base64
import json
from datetime import timedelta
 
import cv2
import numpy as np
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST
from .forms import StudentForm
from .models import Attendance, Student
 
 
# OpenCV付属のHaar Cascadeを使用するので、xmlを別途ダウンロードする必要はありません。
import os

# # ダウンロードしたファイルの絶対パスを直接指定します
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# xml_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# FACE_CASCADE = cv2.CascadeClassifier(xml_path)

 
# #  # OpenCV付属のHaar Cascadeを使用
# # FACE_CASCADE = cv2.CascadeClassifier(
# #     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# # )

# if FACE_CASCADE.empty():
#     raise RuntimeError("Haar Cascadeの読み込みに失敗しました")
 
# OpenCVの顔検出用XML
xml_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'haarcascade_frontalface_default.xml'
)

FACE_CASCADE = cv2.CascadeClassifier(xml_path)

if FACE_CASCADE.empty():
    raise RuntimeError("Haar Cascadeの読み込みに失敗しました")

def top(request):
    return render(request, "attendance/top.html")
 
 
def register_student(request):
    if request.method == "POST":
        form = StudentForm(request.POST, request.FILES)
        if form.is_valid():
            student = form.save()
            messages.success(request, f"{student.name} さんを登録しました。")
            return redirect("attendance:top")
    else:
        form = StudentForm()
 
    return render(request, "attendance/register.html", {"form": form})
 
 
def camera(request):
    return render(request, "attendance/camera.html")
 
 
def _read_student_face(student):
    """登録写真から最大の顔1つを取り出して、200x200のグレースケールにする。"""
    image = cv2.imread(student.photo.path)
    if image is None:
        return None
 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
 
    if len(faces) == 0:
        return None
 
    # 一番大きい顔を使う
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y+h, x:x+w]
    return cv2.resize(face, (200, 200))

def _build_recognizer():
    """有効な園児の登録写真からLBPH認識器を作る。"""
    training_faces = []
    labels = []
    label_to_student_id = {}
 
    students = Student.objects.filter(is_active=True)
 
    label = 0
    for student in students:
        face = _read_student_face(student)
        if face is None:
            continue
 
        training_faces.append(face)
        labels.append(label)
        label_to_student_id[label] = student.id
        label += 1
 
    if not training_faces:
        return None, {}
 
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(training_faces, np.array(labels, dtype=np.int32))
    return recognizer, label_to_student_id
 
 
def _decode_camera_image(data_url):
    """data:image/jpeg;base64,... をOpenCV画像へ変換する。"""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
 
    binary = base64.b64decode(data_url)
    array = np.frombuffer(binary, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)
 
 
def _find_face_from_camera(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
    )
 
    if len(faces) == 0:
        return None
 
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y+h, x:x+w]
    return cv2.resize(face, (200, 200))
 
 
def _next_attendance_type(student):
    """本日最初は登園。すでに登園があり降園がなければ降園。
    すでに降園済みの場合は再登録させない。
    """
    today = timezone.localdate()
    records = Attendance.objects.filter(
        student=student,
        timestamp__date=today,
    )
 
    has_in = records.filter(attendance_type="in").exists()
    has_out = records.filter(attendance_type="out").exists()
 
    if not has_in:
        return "in"
    if not has_out:
        return "out"
    return None
 
 
@require_POST
def recognize_face(request):
    try:
        body = json.loads(request.body)
        data_url = body.get("image")
 
        if not data_url:
            return JsonResponse({"ok": False, "message": "画像がありません。"}, status=400)
 
        image = _decode_camera_image(data_url)
        if image is None:
            return JsonResponse({"ok": False, "message": "画像を読み込めません。"}, status=400)
 
        camera_face = _find_face_from_camera(image)
        if camera_face is None:
            return JsonResponse({
                "ok": False,
                "message": "顔を検出できません。カメラの正面を向いてください。"
            })
 
        recognizer, label_map = _build_recognizer()
        if recognizer is None:
            return JsonResponse({
                "ok": False,
                "message": "認証できる登録写真がありません。先に園児を登録してください。"
            })
 
        label, confidence = recognizer.predict(camera_face)
 
        # LBPHは数値が小さいほど近い。学校デモ用の初期値。
        # 環境によって35〜80程度で調整してください。
        THRESHOLD = 90
        print(f"認証結果: label={label}, confidence={confidence}")
        
        if confidence > THRESHOLD or label not in label_map:
            return JsonResponse({
                "ok": False,
                "message": "登録済みの顔と一致しませんでした。",
                "confidence": round(float(confidence), 2),
            })
 
        student = get_object_or_404(Student, id=label_map[label])
 
        # 数秒以内の連打による二重登録を防ぐ
        recent = Attendance.objects.filter(
            student=student,
            timestamp__gte=timezone.now() - timedelta(seconds=10),
        ).first()
        if recent:
            return JsonResponse({
                "ok": False,
                "message": "連続認証です。少し待ってからもう一度お試しください。"
            })
 
        attendance_type = _next_attendance_type(student)
        if attendance_type is None:
            return JsonResponse({
                "ok": False,
                "message": f"{student.name} さんは本日の登園・降園がすでに完了しています。"
            })
 
        record = Attendance.objects.create(
            student=student,
            attendance_type=attendance_type,
        )
 
        return JsonResponse({
            "ok": True,
            "student": student.name,
            "student_number": student.student_number,
            "type": record.get_attendance_type_display(),
            "time": timezone.localtime(record.timestamp).strftime("%H:%M:%S"),
            "confidence": round(float(confidence), 2),
            "redirect_url": f"/success/{record.id}/",
        })
 
    except Exception as e:
        return JsonResponse({
            "ok": False,
            "message": f"エラーが発生しました: {str(e)}"
        }, status=500)
 
 
def success(request, record_id):
    record = get_object_or_404(Attendance, id=record_id)
    return render(request, "attendance/success.html", {"record": record})
 
 
def dashboard(request):
    today = timezone.localdate()
    students = Student.objects.filter(is_active=True).order_by("student_number")
    rows = []
 
    for student in students:
        records = Attendance.objects.filter(
            student=student,
            timestamp__date=today,
        )
        in_record = records.filter(attendance_type="in").order_by("timestamp").first()
        out_record = records.filter(attendance_type="out").order_by("timestamp").first()
 
        rows.append({
            "student": student,
            "in_record": in_record,
            "out_record": out_record,
        })
 
    return render(
        request,
        "attendance/dashboard.html",
        {"rows": rows, "today": today},
    )


# Create your views here.
