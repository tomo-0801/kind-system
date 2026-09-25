import base64
import json
import os
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

# --- Haar Cascadeの初期化 ---
xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml')
FACE_CASCADE = cv2.CascadeClassifier(xml_path)

if FACE_CASCADE.empty():
    raise RuntimeError("Haar Cascadeの読み込みに失敗しました")


# --- ヘルパー関数 ---
def _get_face(image, min_size):
    """画像から最大の顔を1つ抽出し、200x200のグレースケールで返す"""
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
    
    if len(faces) == 0:
        return None
    
    # 面積(w * h)が最大の顔を取得
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return cv2.resize(gray[y:y+h, x:x+w], (200, 200))


def _build_recognizer():
    """有効な園児の登録写真からLBPH認識器を生成"""
    faces, labels, label_map = [], [], {}
    
    for student in Student.objects.filter(is_active=True):
        face = _get_face(cv2.imread(student.photo.path), min_size=(80, 80))
        if face is not None:
            label = len(faces)
            faces.append(face)
            labels.append(label)
            label_map[label] = student.id

    if not faces:
        return None, {}
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    return recognizer, label_map


def _error_res(msg, status=200, **kwargs):
    """エラー用JsonResponseジェネレータ"""
    return JsonResponse({"ok": False, "message": msg, **kwargs}, status=status)


# --- ページ描画ビュー ---
def top(request):
    return render(request, "attendance/top.html")


def camera(request):
    return render(request, "attendance/camera.html")


def register_student(request):
    # GETとPOSTのフォーム初期化を1行に集約
    form = StudentForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        student = form.save()
        messages.success(request, f"{student.name} さんを登録しました。")
        return redirect("attendance:top")
    
    return render(request, "attendance/register.html", {"form": form})


def success(request, record_id):
    record = get_object_or_404(Attendance, id=record_id)
    return render(request, "attendance/success.html", {"record": record})


def dashboard(request):
    today = timezone.localdate()
    students = Student.objects.filter(is_active=True).order_by("student_number")
    
    # Prefetchの代わりに辞書を用いたN+1問題の回避策（related_nameに依存しない安全な方法）
    # 1回のクエリで本日の全打刻データを取得
    today_records = Attendance.objects.filter(timestamp__date=today).order_by("timestamp")
    
    # 園児IDをキーにした辞書にグループ化
    records_by_student = {}
    for r in today_records:
        records_by_student.setdefault(r.student_id, []).append(r)

    # 辞書からデータを取り出すことで、ループ内のDBアクセス(N+1)をゼロにする
    rows = [
        {
            "student": student,
            "in_record": next((r for r in records_by_student.get(student.id, []) if r.attendance_type == "in"), None),
            "out_record": next((r for r in records_by_student.get(student.id, []) if r.attendance_type == "out"), None),
        }
        for student in students
    ]
    
    return render(request, "attendance/dashboard.html", {"rows": rows, "today": today})


# --- APIビュー ---
@require_POST
def recognize_face(request):
    try:
        data_url = json.loads(request.body).get("image", "")
        if not data_url:
            return _error_res("画像がありません。", status=400)

        # 1度しか使われないBase64デコード処理は関数化せずインラインで簡潔に記述
        binary = base64.b64decode(data_url.split(",", 1)[1] if "," in data_url else data_url)
        image = cv2.imdecode(np.frombuffer(binary, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return _error_res("画像を読み込めません。", status=400)

        face = _get_face(image, min_size=(100, 100))
        if face is None:
            return _error_res("顔を検出できません。カメラの正面を向いてください。")

        recognizer, label_map = _build_recognizer()
        if not recognizer:
            return _error_res("認証できる登録写真がありません。先に園児を登録してください。")

        label, confidence = recognizer.predict(face)
        print(f"認証結果: label={label}, confidence={confidence}")

        if confidence > 90 or label not in label_map:
            return _error_res("登録済みの顔と一致しませんでした。", confidence=round(float(confidence), 2))

        student = get_object_or_404(Student, id=label_map[label])

        # 連打防止 (10秒以内の履歴が存在するかチェック)
        if Attendance.objects.filter(student=student, timestamp__gte=timezone.now() - timedelta(seconds=10)).exists():
            return _error_res("連続認証です。少し待ってからもう一度お試しください。")

        # 本日の打刻種別を判定 (関数化せず、setを用いた高速なインライン判定に統合)
        today_types = set(
            Attendance.objects.filter(student=student, timestamp__date=timezone.localdate())
            .values_list("attendance_type", flat=True)
        )
        attendance_type = "in" if "in" not in today_types else ("out" if "out" not in today_types else None)
        
        if not attendance_type:
            return _error_res(f"{student.name} さんは本日の登園・降園がすでに完了しています。")

        record = Attendance.objects.create(student=student, attendance_type=attendance_type)

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
        return _error_res(f"エラーが発生しました: {str(e)}", status=500)

# --- 園児一覧画面 ---
def student_list(request):
    # 有効な園児一覧を取得
    students = Student.objects.filter(is_active=True).order_by("student_number")
    return render(request, "attendance/student_list.html", {"students": students})

# --- 園児の無効化（卒園・退園処理） ---
@require_POST
def delete_student(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    # 物理削除ではなく is_active を False に変更（過去の打刻データ保護のため）
    student.is_active = False
    student.save()
    messages.success(request, f"{student.name} さんを非表示（退園・卒園）にしました。")
    return redirect("attendance:student_list")