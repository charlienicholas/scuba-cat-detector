import cv2
import os
import time
import sys
import numpy as np
import subprocess 

VIDEO_PATH = "scuba.mp4"
REQUIRED_MOTION_FRAMES = 5 
MOUTH_COVER_SENSITIVITY = 12 
MOUTH_TEXTURE_DROP = 4       
MOTION_SENSITIVITY = 5       


HAS_AUDIO = False
try:
    from ffpyplayer.player import MediaPlayer
    HAS_AUDIO = True
except ImportError:
    pass

def play_scuba_video():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video tidak ditemukan di path:\n{VIDEO_PATH}")
        return

    print("\nAiihhh, saatnya memutar video...")


    try:
        if sys.platform.startswith('linux'):
   
            if os.system("which mpv > /dev/null 2>&1") == 0:
                print("[INFO] Memutar dengan MPV Player (Mode Jendela).")
                subprocess.run(["mpv", VIDEO_PATH]) 
                time.sleep(1) # Jeda agar transisi kembali ke kamera mulus
                return
            
            # Memutar dengan 'vlc' dalam mode Window
            elif os.system("which vlc > /dev/null 2>&1") == 0:
                print("[INFO] Memutar dengan VLC Player (Mode Jendela).")
                subprocess.run(["vlc", "--play-and-exit", VIDEO_PATH])
                time.sleep(1)
                return
    except Exception as e:
        print(f"[WARN] Gagal meluncurkan eksternal player: {e}")

    # --- FALLBACK: Jika tidak punya mpv/vlc, gunakan OpenCV ---
    print("[WARN] Memutar dengan OpenCV. Suara mungkin tidak muncul di Fedora/Linux.")
    cap_video = cv2.VideoCapture(VIDEO_PATH)
    fps = cap_video.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    player = MediaPlayer(VIDEO_PATH) if HAS_AUDIO else None
    window_name = "SCUBA MODE - Tekan Q untuk tutup"
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 450)
    
    while cap_video.isOpened():
        ret, frame = cap_video.read()
        if not ret: break
        
        if HAS_AUDIO and player is not None:
            audio_frame, val = player.get_frame()
            if val == 'eof':
                break
                
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'): 
            break
            
    cap_video.release()
    if HAS_AUDIO and player is not None:
        player.close_player()
    cv2.destroyWindow(window_name)
    cv2.waitKey(100) 

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak dapat dibuka. Pastikan tidak sedang dipakai aplikasi lain.")
        return

    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Feed', 800, 600)

    print("\n" + "="*50)
    print(" PROGRAM SCUBA CAT AKTIF (MODE OPENCV MURNI) ")
    print("="*50)
    print("1. Tangan KIRI: Tutup area mulut Anda.")
    print("2. Tangan KANAN: Gerakkan/lambai di sebelah kanan.")
    print("Tekan ESC untuk keluar dari program.")
    print("="*50 + "\n")

    motion_count = 0
    
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    prev_gray = None
    baseline_mouth_mean = None
    baseline_mouth_std = None
    
    last_face = None
    face_missing_frames = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: 
            print("Gagal membaca frame, mencoba lagi...")
            time.sleep(0.1)
            continue
            
        image = cv2.flip(image, 1) 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))
        is_triggered = False
        left_hand_on_mouth = False
        is_moving = False
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            last_face = largest_face
            face_missing_frames = 0
        else:
            face_missing_frames += 1

        if last_face is not None and face_missing_frames < 15:
            x, y, w, h = last_face
            
            box_color = (255, 0, 0) if face_missing_frames == 0 else (100, 100, 100)
            cv2.rectangle(image, (x, y), (x+w, y+h), box_color, 2)
            
            mouth_roi = gray[y + int(h*0.6):y + h, x:x + w]
            
            if mouth_roi.shape[0] > 0 and mouth_roi.shape[1] > 0:
                current_mouth_mean = np.mean(mouth_roi)
                current_mouth_std = np.std(mouth_roi) 
                
                cv2.rectangle(image, (x, y + int(h*0.6)), (x+w, y+h), (0, 255, 0), 1)
                
                if baseline_mouth_mean is None:
                    baseline_mouth_mean = current_mouth_mean
                    baseline_mouth_std = current_mouth_std
                
                # Hitung perbedaan kecerahan
                mean_diff = abs(current_mouth_mean - baseline_mouth_mean)
                
                # Hitung penurunan tekstur (Jika ditutup tangan, tekstur bibir/gigi hilang, nilai std akan turun)
                std_drop = baseline_mouth_std - current_mouth_std 
                
                # TRIGGER: Jika kecerahan berubah drastis ATAU tekstur menghilang dengan signifikan
                if mean_diff > MOUTH_COVER_SENSITIVITY or std_drop > MOUTH_TEXTURE_DROP: 
                    left_hand_on_mouth = True
                else:
                    baseline_mouth_mean = current_mouth_mean * 0.15 + baseline_mouth_mean * 0.85
                    baseline_mouth_std = current_mouth_std * 0.15 + baseline_mouth_std * 0.85
                    
            right_roi_x = min(x + w + 20, image.shape[1]-1)
            right_roi = gray[0:image.shape[0], right_roi_x:image.shape[1]]
            cv2.rectangle(image, (right_roi_x, 0), (image.shape[1], image.shape[0]), (0, 255, 255), 1)
            
            if prev_gray is not None and right_roi.shape[1] > 0:
                prev_right_roi = prev_gray[0:image.shape[0], right_roi_x:image.shape[1]]
                diff = cv2.absdiff(right_roi, prev_right_roi)
                motion_level = np.mean(diff)
                if motion_level > MOTION_SENSITIVITY: 
                    is_moving = True
                    
        prev_gray = gray.copy()
        
        color_left = (0, 255, 0) if left_hand_on_mouth else (0, 0, 255)
        cv2.putText(image, f"Tutup Mulut: {'OK' if left_hand_on_mouth else 'BELUM'}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_left, 2)
        
        color_right = (0, 255, 0) if is_moving else (0, 0, 255)
        cv2.putText(image, f"Kanan Gerak: {'OK' if is_moving else 'BELUM'}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_right, 2)
        
        if left_hand_on_mouth and is_moving:
            motion_count += 1
            cv2.putText(image, "MEMPROSES...", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            motion_count = max(0, motion_count - 1)

        if motion_count >= REQUIRED_MOTION_FRAMES:
            is_triggered = True
            motion_count = 0
            
            baseline_mouth_mean = None 
            baseline_mouth_std = None
                
        cv2.imshow('Camera Feed', image)

        if is_triggered:
            cv2.rectangle(image, (0,0), (image.shape[1], image.shape[0]), (0, 255, 0), 10)
            cv2.imshow('Camera Feed', image)
            cv2.waitKey(200)
            
            play_scuba_video()
            
            motion_count = 0
            prev_gray = None
            last_face = None

        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()