# 🚗 License Plate Detection using OpenCV & Tesseract OCR

A Python-based computer vision project that detects and reads vehicle license plate numbers from images using OpenCV and Tesseract OCR. Built for real-world traffic scenes with Haar cascades, contour fallback, and adaptive image preprocessing.

---

## 📸 Sample Output

> Input:
> `images/test_plate.jpg`

> Output:  
> `📄 License Plate Text: Aa SraaeR` *(example from test image)*  
> Cropped plate image saved to `output/test_plate_plate.png`

---

## 🧠 Features

- License plate detection using OpenCV Haar cascade
- Edge detection + contour fallback if Haar fails
- OCR reading using Tesseract (PSM 8)
- Adaptive preprocessing (blurring, thresholding, Canny)
- CLI interface for flexible input
- Output: printed text + saved cropped image

---

## 🛠 Tech Stack

- `OpenCV` for image processing
- `pytesseract` for OCR
- `NumPy` for image matrix operations

---

## 📁 Project Structure

