import cv2
import numpy as np
# import imutils
from imutils.object_detection import non_max_suppression # để loại bỏ chồng lấn

filename = 'video/input/walking.mp4'
filesize = (1920, 1080)
scale_ratio = 1 # tỉ lệ phóng đại kích thước vid khi cần thiết

# lưu vid đầu ra
output_filename = 'video/output/walking.mp4'
output_frames_per_second = 20.0

def main():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # load vid
    cap = cv2.VideoCapture(filename)

    # kiểm tra xem vid có được mở không
    if not cap.isOpened:
        print('Error: the video is not opened')
        return
    
    # tạo đối tượng VideoWriter để lưu vid đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(output_filename, fourcc, output_frames_per_second, filesize)

    # xử lý vid
    while cap.isOpened():
        # đọc từn frame
        success, frame = cap.read()

        if not success:
            print('Error: the video is not opened')
            break
        else:
            # thay đổi kích thước khung hình
            width = int(frame.shape[1] * scale_ratio)
            height = int(frame.shape[0] * scale_ratio)
            frame = cv2.resize(frame, (width, height))

            # lưu frame ban đầu
            orig_frame = frame.copy()

            # phát hiện người trong frame
            bounding_boxes, weights = hog.detectMultiScale(frame, winStride=(16, 16), padding=(4, 4), scale=1.05)

            for(x, y, w, h) in bounding_boxes:
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # loại bỏ chồng lấn
            # Thay đổi chỉ số overlapThresh để được kết quả tốt nhất
            bounding_boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in bounding_boxes])
            selection = non_max_suppression(bounding_boxes, probs=None, overlapThresh=0.45)

            # vẽ bounding box cuối cùng
            for(x1, y1, x2, y2) in selection:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

            # ghi frame vào vid đầu ra
            result.write(frame)

            # hiển thị frame
            cv2.imshow('frame', frame)

            # hiển thị khung hình trong x mili giây và quit nếu nhấn 'q'
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # dừng lại khi vid kết thúc
    cap.release()

    # giải phóng quá trình ghi vid
    result.release()

    # đóng tất cả cửa sổ
    cv2.destroyAllWindows()

main()