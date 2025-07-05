import cv2
import pytesseract
import os

# Optional: Set path if tesseract is not in PATH
pytesseract.pytesseract.tesseract_cmd = r"D:\Kaam ki chize\tesseract.exe"

def preprocess_plate(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 100, 200)
    return edged

def detect_and_read_plate(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Image not found")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar cascade
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(plates) == 0:
        print("⚠️ No plates detected with Haar cascade. Trying fallback method...")
        fallback_contour_detection(image)
        return

    for (x, y, w, h) in plates:
        roi = image[y:y+h, x:x+w]
        processed = preprocess_plate(roi)

        # OCR
        text = pytesseract.image_to_string(processed, config="--psm 8 --oem 3")
        print(f"📄 License Plate Text: {text.strip()}")

        # Save output
        os.makedirs("output", exist_ok=True)
        out_path = f"output/{os.path.basename(image_path).split('.')[0]}_plate.png"
        cv2.imwrite(out_path, roi)

        # Draw results
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(image, text.strip(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("Detected Plate", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fallback_contour_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        if 2 < aspect_ratio < 6 and w > 100 and h > 20:
            roi = image[y:y+h, x:x+w]
            processed = preprocess_plate(roi)
            text = pytesseract.image_to_string(processed, config="--psm 8 --oem 3")

            if len(text.strip()) >= 5:
                print(f"🆗 Fallback OCR Result: {text.strip()}")
                cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Fallback Contour Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = input("Enter path to vehicle image (e.g., images/car1.jpg): ")
    detect_and_read_plate(img_path)
