import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  # Added ImageTk import
import cv2 as cv
import numpy as np
from functions import (
    adjustBrightness, contrastDetection, blur, detectingBarCode,
    decode_barcode, noiseDetection, rotationDetection, sharpen_if_needed,
    objectDetection, barcodeErosion
)


# Function to load and process the image
def process_image():
    global img
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.bmp")]
    )
    if not file_path:
        return

    # Read and display the selected image
    img = cv.imread(file_path, 0)
    if img is None:
        messagebox.showerror("Error", "Failed to load image.")
        return

    # Update the GUI to display the original image
    display_image(img, "Original Image", original_canvas)

    # Process the image
    try:
        processed_img = adjustBrightness(img)
        processed_img, flag_2 = contrastDetection(processed_img)
        if flag_2 == 2:
            blurred_img = blur(processed_img)
            cropped_barcode = detectingBarCode(blurred_img, flag_2)
            decoded_value = decode_barcode(cropped_barcode)
        else:
            processed_img = noiseDetection(processed_img)
            processed_img, flag = rotationDetection(processed_img, 0)
            processed_img = sharpen_if_needed(processed_img)
            processed_img, flag = objectDetection(processed_img, flag)
            if flag == 1:
                _, processed_img = cv.threshold(processed_img, 128, 255, cv.THRESH_BINARY)
                cropped_barcode = detectingBarCode(processed_img, 0)
                processed_img = barcodeErosion(cropped_barcode)
                kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 200))
                processed_img = cv.morphologyEx(processed_img, cv.MORPH_CLOSE, kernel)
                processed_img = cv.dilate(processed_img, kernel, iterations=2)
                decoded_value = decode_barcode(processed_img)
            else:
                cropped_barcode = detectingBarCode(processed_img, 0)
                decoded_value = decode_barcode(cropped_barcode)

        # Update the GUI to display the processed image
        display_image(processed_img, "Processed Image", processed_canvas)

        # Display the decoded barcode value
        result_label.config(text=f"Decoded Barcode: {decoded_value}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image:\n{e}")


# Function to display images in the GUI
def display_image(img, title, canvas):
    img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB) if len(img.shape) == 2 else cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil.resize((600, 450), Image.Resampling.LANCZOS))  # Updated resize dimensions
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk


# Create the main window
root = tk.Tk()
root.title("Barcode Detection GUI")
root.geometry("1000x800")  # Increased the size of the main window

# Create frames for the layout
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

middle_frame = tk.Frame(root)
middle_frame.pack()

bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=10)

# Add buttons
upload_button = tk.Button(top_frame, text="Upload Image", command=process_image, width=20)
upload_button.pack()

# Add canvases for displaying images with larger size
original_canvas = tk.Canvas(middle_frame, width=600, height=450, bg="white")  # Increased canvas size
original_canvas.grid(row=0, column=0, padx=10, pady=10)
processed_canvas = tk.Canvas(middle_frame, width=600, height=450, bg="white")  # Increased canvas size
processed_canvas.grid(row=0, column=1, padx=10, pady=10)

# Add labels
original_label = tk.Label(middle_frame, text="Original Image")
original_label.grid(row=1, column=0, pady=5)
processed_label = tk.Label(middle_frame, text="Processed Image")
processed_label.grid(row=1, column=1, pady=5)

# Add a result label
result_label = tk.Label(bottom_frame, text="Decoded Barcode: ", font=("Arial", 14))
result_label.pack()

# Start the GUI event loop
root.mainloop()

