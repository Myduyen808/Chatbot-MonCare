"""
audio_utils.py
Module tiền xử lý âm thanh: Chuyển tiếng khóc thành văn bản (Text)
Hỗ trợ cả đầu vào là Bytes (từ Micro) và File Path (từ Upload)
Tích hợp bộ lọc nhiễu (Denoising) để xử lý âm thanh từ micro web.
"""
import io
import librosa
import noisereduce as nr
import torch
from transformers import pipeline
import numpy as np

# Cache model để không tải lại mỗi lần gọi
_cry_classifier = None

def get_cry_classifier():
    global _cry_classifier
    if _cry_classifier is None:
        print("⏳ Đang tải model phân tích tiếng khóc (lần đầu sẽ mất chút thời gian)...")
        _cry_classifier = pipeline(
            "audio-classification", 
            model="MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        print("✅ Tải model thành công!")
    return _cry_classifier

def preprocess_audio(audio_bytes):
    """
    Lọc nhiễu (Denoising) cho âm thanh thu từ micro của trình duyệt.
    """
    try:
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        if len(audio_array.shape) > 1:
            audio_array = librosa.to_mono(audio_array)
            
        clean_audio = nr.reduce_noise(y=audio_array, sr=sr, prop_decrease=0.7)
        
        return clean_audio, sr
    except Exception as e:
        print(f"⚠️ Lỗi khi lọc nhiễu (sẽ dùng âm thanh gốc): {e}")
        return None, None


def classify_cry_reason(audio_input, confidence):
    """
    Phân tích chính xác dựa trên:
    - KHÓC ĐÓI: Pitch 400-800Hz, Rhythm LÊN XUỐNG rõ, ZCR trung bình
    - KHÓC ĐAU: Pitch RẤT CAO >800Hz, Energy CAO, ZCR CAO, liên tục
    - KHÓC BUỒN NGỦ: Pitch THẤP <350Hz, Energy THẤP, ZCR THẤP
    - KHÓC KHÓ CHỊU: Pitch 350-600Hz, ngắt quãng rõ
    - KHÓC NHIỆT ĐỘ: Pitch trung bình, Rhythm ĐỀU, kéo dài
    """
    try:
        if isinstance(audio_input, bytes):
            processed_audio, sr = preprocess_audio(audio_input)
            if processed_audio is None:
                processed_audio, sr = librosa.load(io.BytesIO(audio_input), sr=16000)
        else:
            processed_audio, sr = librosa.load(audio_input, sr=16000)
        
        # 1. THỜI LƯỢNG
        duration_sec = len(processed_audio) / sr
        
        # 2. CƯỜNG ĐỘ (RMS Energy)
        rms_energy = float(np.sqrt(np.mean(processed_audio**2)))
        
        # 3. PHÂN TÍCH PITCH (dùng pyin thay piptrack)
        try:
            f0, voiced_flag, _ = librosa.pyin(
                processed_audio, 
                fmin=librosa.note_to_hz('C2'),  # ~65Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093Hz
                sr=sr
            )
            valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
            
            if len(valid_f0) > 0:
                avg_pitch = float(np.mean(valid_f0))
                max_pitch = float(np.max(valid_f0))
                pitch_std = float(np.std(valid_f0))
            else:
                avg_pitch = max_pitch = pitch_std = 0
        except:
            avg_pitch = max_pitch = pitch_std = 0

        # 4. ZERO-CROSSING RATE (ZCR)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=processed_audio)[0]))
        
        # 5. SPECTRAL CENTROID
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=processed_audio, sr=sr)[0]))
        
        # 6. PHÂN TÍCH RHYTHM
        frame_length = int(0.25 * sr)
        num_frames = len(processed_audio) // frame_length
        frame_energies = []
        
        if num_frames > 2:
            for i in range(num_frames):
                frame = processed_audio[i*frame_length:(i+1)*frame_length]
                frame_energy = float(np.sqrt(np.mean(frame**2)))
                frame_energies.append(frame_energy)
            
            energy_threshold = np.mean(frame_energies) * 0.6
            cry_frames = [e for e in frame_energies if e > energy_threshold]
            cry_ratio = len(cry_frames) / len(frame_energies)
            
            if len(frame_energies) > 1:
                energy_changes = np.diff(frame_energies)
                energy_variability = float(np.std(energy_changes))
                transitions = 0
                for i in range(1, len(frame_energies)):
                    if frame_energies[i-1] <= energy_threshold and frame_energies[i] > energy_threshold:
                        transitions += 1
                cry_bursts = transitions
            else:
                energy_variability = 0
                cry_bursts = 0
        else:
            cry_ratio = 0
            energy_variability = 0
            cry_bursts = 0
            
        # 7. XU HƯỚNG ENERGY
        if len(frame_energies) > 4:
            first_half = np.mean(frame_energies[:len(frame_energies)//2])
            second_half = np.mean(frame_energies[len(frame_energies)//2:])
            energy_trend = second_half - first_half
        else:
            energy_trend = 0

        # 8. HỆ THỐNG SCORING
        scores = {"hunger": 0, "pain": 0, "fatigue": 0, "discomfort": 0, "temperature": 0}

        # --- KHÓC ĐÓI ---
        if avg_pitch > 0:
            if 400 <= avg_pitch <= 800: scores["hunger"] += 3
            elif 300 <= avg_pitch <= 400: scores["hunger"] += 1
        if cry_bursts >= 3: scores["hunger"] += 3
        elif cry_bursts >= 2: scores["hunger"] += 2
        elif cry_bursts >= 1: scores["hunger"] += 1
        if avg_pitch > 0 and 40 <= pitch_std <= 150: scores["hunger"] += 2
        elif avg_pitch > 0 and pitch_std > 150: scores["hunger"] += 1
        if energy_trend > 0.005: scores["hunger"] += 2
        elif energy_trend > 0: scores["hunger"] += 1
        if 0.01 <= energy_variability <= 0.06: scores["hunger"] += 1
        if 0.05 <= zcr <= 0.15: scores["hunger"] += 1

        # --- KHÓC ĐAU ---
        if avg_pitch > 0:
            if avg_pitch > 800: scores["pain"] += 4
            elif avg_pitch > 600: scores["pain"] += 2
            if max_pitch > 1500: scores["pain"] += 2
            elif max_pitch > 1000: scores["pain"] += 1
        if rms_energy > 0.2: scores["pain"] += 3
        elif rms_energy > 0.15: scores["pain"] += 2
        elif rms_energy > 0.1: scores["pain"] += 1
        if cry_ratio > 0.9: scores["pain"] += 3
        elif cry_ratio > 0.8: scores["pain"] += 2
        if cry_bursts <= 1 and cry_ratio > 0.7: scores["pain"] += 2
        if zcr > 0.15: scores["pain"] += 2
        elif zcr > 0.1: scores["pain"] += 1
        if spectral_centroid > 3000: scores["pain"] += 2
        elif spectral_centroid > 2000: scores["pain"] += 1
        if energy_variability < 0.02 and rms_energy > 0.1: scores["pain"] += 1

        # --- KHÓC BUỒN NGỦ ---
        if avg_pitch > 0:
            if avg_pitch < 300: scores["fatigue"] += 3
            elif avg_pitch < 350: scores["fatigue"] += 2
            elif avg_pitch < 400: scores["fatigue"] += 1
        if rms_energy < 0.04: scores["fatigue"] += 3
        elif rms_energy < 0.06: scores["fatigue"] += 2
        elif rms_energy < 0.08: scores["fatigue"] += 1
        if zcr < 0.04: scores["fatigue"] += 2
        elif zcr < 0.06: scores["fatigue"] += 1
        if spectral_centroid < 1500: scores["fatigue"] += 2
        elif spectral_centroid < 2000: scores["fatigue"] += 1
        if avg_pitch > 0 and pitch_std < 30: scores["fatigue"] += 2
        elif avg_pitch > 0 and pitch_std < 50: scores["fatigue"] += 1
        if cry_ratio < 0.4: scores["fatigue"] += 2
        elif cry_ratio < 0.55: scores["fatigue"] += 1
        if energy_trend < -0.005: scores["fatigue"] += 2
        elif energy_trend < -0.002: scores["fatigue"] += 1

        # --- KHÓC KHÓ CHỊU ---
        if avg_pitch > 0 and 300 <= avg_pitch <= 550: scores["discomfort"] += 1
        if 0.3 <= cry_ratio <= 0.6: scores["discomfort"] += 3
        elif 0.6 <= cry_ratio <= 0.75: scores["discomfort"] += 1
        if cry_bursts >= 4: scores["discomfort"] += 2
        elif cry_bursts >= 3: scores["discomfort"] += 1
        if energy_variability > 0.05: scores["discomfort"] += 2
        elif energy_variability > 0.03: scores["discomfort"] += 1
        if 0.03 <= rms_energy <= 0.1: scores["discomfort"] += 1

        # --- KHÓC NHIỆT ĐỘ ---
        if avg_pitch > 0 and 400 <= avg_pitch <= 650: scores["temperature"] += 1
        if energy_variability < 0.025 and cry_ratio > 0.5: scores["temperature"] += 2
        elif energy_variability < 0.035: scores["temperature"] += 1
        if duration_sec > 8: scores["temperature"] += 2
        elif duration_sec > 5: scores["temperature"] += 1
        if avg_pitch > 0 and pitch_std < 40: scores["temperature"] += 1
        
        # ═══════════════════════════════════════════════════
        # 9. XÁC ĐỊNH NGUYÊN NHÂN
        # ═══════════════════════════════════════════════════
        best_reason = max(scores, key=scores.get)
        best_score = scores[best_reason]
        sorted_reasons = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        second_best_score = sorted_reasons[1][1] if len(sorted_reasons) > 1 else 0
        
        if best_score < 4 or (best_score - second_best_score < 2 and best_score < 6):
            best_reason = "unknown"
        
        REASON_MAP_VI = {
            "hunger": "🥛 ĐÓI - Cần cho bú ngay",
            "pain": "😰 ĐAU ĐỚN - Cần kiểm tra ngay",
            "fatigue": "😴 BUỒN NGỦ/THIẾU NGỦ - Cần giúp bé ngủ",
            "discomfort": "🧷 KHÓ CHỊU - Kiểm tra tã/quần áo",
            "temperature": "🌡️ QUÁ NÓNG/LẠNH - Kiểm tra nhiệt độ",
            "unknown": "🔍 CHƯA XÁC ĐỊNH RÕ - Kiểm tra các nguyên nhân phổ biến"
        }
        
        description = f"""**📊 Chi tiết phân tích âm thanh:**

| Chỉ số | Giá trị | Phân loại |
|--------|---------|-----------|
| Thời lượng | {duration_sec:.1f}s | {"Ngắn" if duration_sec < 3 else "TB" if duration_sec < 10 else "Dài"} |
| Cường độ (RMS) | {rms_energy:.4f} | {"Thấp (rên)" if rms_energy < 0.06 else "TB" if rms_energy < 0.12 else "Cao (la)"} |
| Tần số (Pitch) | {avg_pitch:.1f} Hz | {"Thấp" if avg_pitch < 350 else "TB" if avg_pitch < 700 else "Cao"} |
| Biến thiên pitch | {pitch_std:.1f} Hz | {"Đều" if pitch_std < 40 else "Lên xuống" if pitch_std < 120 else "Lớn"} |
| ZCR | {zcr:.4f} | {"Trầm" if zcr < 0.05 else "TB" if zcr < 0.12 else "Gắt"} |
| Spectral Centroid | {spectral_centroid:.0f} Hz | {"Trầm" if spectral_centroid < 2000 else "TB" if spectral_centroid < 3000 else "Sắc"} |
| Tỷ lệ khóc | {cry_ratio*100:.1f}% | {"Ít" if cry_ratio < 0.5 else "TB" if cry_ratio < 0.8 else "Nhiều"} |
| Số cụm khóc | {cry_bursts} | {"1 cụm" if cry_bursts <= 1 else "2-3 cụm" if cry_bursts <= 3 else "Nhiều cụm"} |
| Xu hướng energy | {"Tăng ↗" if energy_trend > 0.003 else "Giảm ↘" if energy_trend < -0.003 else "Ổn định →"} | {"Bé khóc to dần" if energy_trend > 0.003 else "Bé yếu dần" if energy_trend < -0.003 else "Đều"} |

**🎯 Bảng điểm chẩn đoán:**
| Nguyên nhân | Điểm | Thanh điểm |
|-------------|------|------------|
| 🥛 Khóc đói | {scores['hunger']} | {"█" * scores['hunger']}{"░" * (10 - scores['hunger'])} |
| 😰 Khóc đau | {scores['pain']} | {"█" * scores['pain']}{"░" * (10 - scores['pain'])} |
| 😴 Khóc buồn ngủ | {scores['fatigue']} | {"█" * scores['fatigue']}{"░" * (10 - scores['fatigue'])} |
| 🧷 Khóc khó chịu | {scores['discomfort']} | {"█" * scores['discomfort']}{"░" * (10 - scores['discomfort'])} |
| 🌡️ Khóc nhiệt độ | {scores['temperature']} | {"█" * scores['temperature']}{"░" * (10 - scores['temperature'])} |

> **📌 Kết luận:** `{REASON_MAP_VI[best_reason]}` (Điểm cao nhất: {best_score}, chênh lệch: {best_score - second_best_score})"""
        
        return best_reason, REASON_MAP_VI[best_reason], description
        
    except Exception as e:
        print(f"⚠️ Lỗi phân loại nguyên nhân: {e}")
        import traceback
        traceback.print_exc()
        return "unknown", "🔍 CHƯA XÁC ĐỊNH RÕ NGUYÊN NHÂN", None



def analyze_baby_cry(audio_input):
    """
    Phân tích tiếng khóc và trả về NGUYÊN NHÂN cụ thể.
    
    Returns:
        reason: str - mã nguyên nhân (hunger, pain, fatigue, discomfort, temperature, unknown, none)
        reason_vi: str - tên nguyên nhân tiếng Việt
        confidence: float - độ tin cậy từ model
        acoustic_desc: str - mô tả chi tiết đặc điểm âm thanh
    """
    try:
        if isinstance(audio_input, bytes):
            processed_audio, sr = preprocess_audio(audio_input)
            if processed_audio is None:
                processed_audio, sr = librosa.load(io.BytesIO(audio_input), sr=16000)
        else:
            processed_audio, sr = librosa.load(audio_input, sr=16000)

        # ── KIỂM TRA ENERGY TRƯỚC KHI PHÂN TÍCH ──
        rms_check = float(np.sqrt(np.mean(processed_audio**2)))
        print(f"🔈 RMS energy của audio: {rms_check:.4f}")
        if rms_check < 0.005:
            return "none", "🔇 ÂM THANH QUÁ NHỎ - Vui lòng để mic gần hơn và thử lại", 0, None

        classifier = get_cry_classifier()
        predictions = classifier(processed_audio)

        print("=== TẤT CẢ PREDICTIONS ===")
        for p in predictions:
            print(p['label'], p['score'])
        
        cry_label = None
        cry_score = 0.0
        
        CRY_KEYWORDS = ["cry", "infant", "wail", "moan", "whimper", "sob", "scream", "yell"]
        
        for pred in predictions:
            label = pred['label']
            if any(kw in label.lower() for kw in CRY_KEYWORDS):
                if pred['score'] > cry_score:
                    cry_score = pred['score']
                    cry_label = label
        
        if cry_label and cry_score > 0.015:
            print(f"🔊 Phát hiện tiếng khóc: {cry_label} ({cry_score:.2f})")
            reason, reason_vi, acoustic_desc = classify_cry_reason(audio_input, cry_score)
            return reason, reason_vi, cry_score, acoustic_desc
            
        else:
            print("🔊 Không nhận diện được tiếng khóc rõ ràng")
            return "none", "❌ KHÔNG PHÁT HIỆN KHÓC", cry_score, None
            
    except Exception as e:
        print(f"❌ Lỗi khi phân tích âm thanh: {e}")
        import traceback
        traceback.print_exc()
        return "none", "❌ LỖI PHÂN TÍCH", 0, None