import glob
import math
import os
import re
import time
import urllib.request
from tkinter import Tk, Frame, Button, Label, Canvas, Scrollbar
import cv2
import numpy as np
from PIL import Image, ImageTk

# Global variables for glasses selection
selected_glasses_path = None
glasses_window = None
number_of_glasses_per_page = 20


def rotate_image(image, angle):
    # Explicitly convert to integers
    height, width = image.shape[:2]
    center = (width // 2, height // 2)  # Simple tuple of integers
    rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))


# Function to make a white background transparent in the glasses image
def make_white_transparent(image_path):
    # Load the image with alpha channel if it exists
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # If the image doesn't have an alpha channel, add one
    if img.shape[2] < 4:
        # Convert to RGBA
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # Create a binary mask of the white background
        # Adjust these thresholds based on your image
        white_mask = np.all(img > 190, axis=2)

        # Set alpha channel to 0 for white pixels
        tmp[:, :, 3] = np.where(white_mask, 0, 255)
        img = tmp

        # Save the processed image with transparent background
        file_name = os.path.basename(image_path)
        transparent_images = os.path.join(os.path.dirname(image_path), "transparent_images")
        processed_path = os.path.join(transparent_images,f"transparent_{file_name}")
        cv2.imwrite(processed_path, img)
        print(f"Processed transparent glasses saved to {processed_path}")

    return img


def overlay_glasses(frame, face, glasses_img, landmarks=None):
    x, y, w, h = face

    # Initialize static variables for smoothing if they don't exist
    if not hasattr(overlay_glasses, "smooth_eye_x"):
        overlay_glasses.smooth_eye_x = None
        overlay_glasses.smooth_eye_y = None
        overlay_glasses.smooth_angle = 0
        overlay_glasses.smooth_width = 0

    if landmarks is not None and len(landmarks) > 0:
        # Get landmarks for the eyes
        landmarks = landmarks[0]  # First face

        # Facemark LBF gives points as shape (68, 1, 2) - need to reshape
        landmarks = landmarks.reshape(-1, 2)

        # Get left and right eye corners (assuming 68-point model)
        left_eye_points = landmarks[36:42]  # Left eye landmarks
        right_eye_points = landmarks[42:48]  # Right eye landmarks

        # Calculate eye centers
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)

        # Calculate angle
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = math.degrees(math.atan2(dy, dx))

        # Position glasses on face using eye coordinates
        eye_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
        eye_center_y = (left_eye_center[1] + right_eye_center[1]) // 2

        # Adjust width based on interpupillary distance
        eye_width = abs(right_eye_center[0] - left_eye_center[0]) * 2.5

        # Apply simple smoothing
        smooth_factor = 0.7  # Adjust between 0-1 (higher = more smoothing)

        # Initialize smoothed values if None
        if overlay_glasses.smooth_eye_x is None:
            overlay_glasses.smooth_eye_x = eye_center_x
            overlay_glasses.smooth_eye_y = eye_center_y
            overlay_glasses.smooth_angle = angle
            overlay_glasses.smooth_width = eye_width
        else:
            # Apply smoothing
            overlay_glasses.smooth_eye_x = int(
                overlay_glasses.smooth_eye_x * smooth_factor + eye_center_x * (1 - smooth_factor))
            overlay_glasses.smooth_eye_y = int(
                overlay_glasses.smooth_eye_y * smooth_factor + eye_center_y * (1 - smooth_factor))
            overlay_glasses.smooth_angle = overlay_glasses.smooth_angle * smooth_factor + angle * (1 - smooth_factor)
            overlay_glasses.smooth_width = overlay_glasses.smooth_width * smooth_factor + eye_width * (
                    1 - smooth_factor)

        # Use smoothed values
        eye_h = int(h * 0.35)  # Height of glasses
        eye_w = int(overlay_glasses.smooth_width)  # Width based on smoothed eye distance
        eye_x = int(overlay_glasses.smooth_eye_x - eye_w / 2)
        eye_y = int(overlay_glasses.smooth_eye_y - eye_h / 2)
        angle = overlay_glasses.smooth_angle
    else:
        # Fallback to original method if no landmarks
        eye_y = y + int(h * 0.28)  # Position for eyes
        eye_h = int(h * 0.35)  # Height of glasses
        eye_w = w  # Width of glasses
        eye_x = max(0, x - int(w * 0.05))  # Center glasses
        angle = 0  # No rotation

    # Safety check for dimensions
    if eye_h <= 0 or eye_w <= 0:
        return frame

    # Resize glasses to fit face
    try:
        resized_glasses = cv2.resize(glasses_img, (eye_w, eye_h))
    except Exception as e:
        print(f"Resize error: {e}")
        return frame

    # Only proceed if we have alpha channel
    if resized_glasses.shape[2] != 4:
        return frame

    # Rotate glasses if we have an angle
    if angle != 0:
        resized_glasses = rotate_image(resized_glasses, angle)

    # Get alpha channel
    alpha_mask = resized_glasses[:, :, 3] / 255.0

    # Ensure we stay within frame boundaries
    roi_y = max(0, min(eye_y, frame.shape[0] - 1))
    roi_h = min(resized_glasses.shape[0], frame.shape[0] - roi_y)
    roi_x = max(0, min(eye_x, frame.shape[1] - 1))
    roi_w = min(resized_glasses.shape[1], frame.shape[1] - roi_x)

    # Skip if ROI is invalid
    if roi_h <= 0 or roi_w <= 0:
        return frame

    # For each color channel, blend the image
    for c in range(0, 3):
        # Get ROI for this operation
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, c]

        # Get corresponding part of glasses and alpha
        glasses_roi = resized_glasses[:roi_h, :roi_w, c]
        alpha_roi = alpha_mask[:roi_h, :roi_w]

        # Apply blending
        frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, c] = (
                roi * (1 - alpha_roi) + glasses_roi * alpha_roi
        )

    return frame


def select_glasses(glasses_path):
    global selected_glasses_path
    selected_glasses_path = glasses_path
    print(f"Selected glasses: {glasses_path}")


def create_ui(glasses_dir):
    global glasses_window
    global number_of_glasses_per_page

    # Create main Tkinter window
    glasses_window = Tk()
    glasses_window.title("Virtual Eyewear Try-on")
    glasses_window.geometry("800x600")

    # Create frame for video display (top portion)
    video_frame = Frame(glasses_window, bg="black")
    video_frame.pack(side="top", fill="both", expand=True)

    # Create label for video display
    video_label = Label(video_frame)
    video_label.pack(pady=10)

    # Create container frame for horizontal scroll
    scroll_container = Frame(glasses_window)
    scroll_container.pack(side="bottom", fill="x", pady=10)

    # Create navigation button frame
    nav_frame = Frame(scroll_container)
    nav_frame.pack(side="top", fill="x", pady=5)

    # Previous button
    prev_button = Button(nav_frame, text="< Previous", state="disabled")
    prev_button.pack(side="left", padx=10)

    # Next button
    next_button = Button(nav_frame, text="Next >")
    next_button.pack(side="right", padx=10)

    # Initialize page info label
    page_info = Label(nav_frame, text="Loading images...")
    page_info.pack(side="top", pady=5)

    # Create a frame for glasses display with proper scrolling
    glasses_scroll_frame = Frame(scroll_container)
    glasses_scroll_frame.pack(side="top", fill="x", expand=True, pady=5)

    # Create canvas for scrolling
    scroll_canvas = Canvas(glasses_scroll_frame, height=150)
    scroll_canvas.pack(side="top", fill="x", expand=True)

    # Add horizontal scrollbar
    scrollbar = Scrollbar(glasses_scroll_frame, orient="horizontal", command=scroll_canvas.xview)
    scrollbar.pack(side="bottom", fill="x")
    scroll_canvas.configure(xscrollcommand=scrollbar.set)

    # Create a frame inside canvas for the glasses items
    glasses_frame = Frame(scroll_canvas)
    canvas_window = scroll_canvas.create_window((0, 0), window=glasses_frame, anchor="nw")

    # Configure the canvas to resize with the frame
    def configure_scroll_region(event):
        # Update the scroll region to encompass the inner frame
        scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

    glasses_frame.bind('<Configure>', configure_scroll_region)

    # Enable mousewheel scrolling
    def _on_mousewheel(event):
        scroll_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # Also bind shift+mousewheel for horizontal scrolling (common in many systems)
    def _on_shift_mousewheel(event):
        if event.state & 0x1:  # Check if Shift is pressed
            scroll_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    scroll_canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)

    # Store components in window for access
    glasses_window.glasses_frame = glasses_frame
    glasses_window.prev_button = prev_button
    glasses_window.next_button = next_button
    glasses_window.page_info = page_info
    glasses_window.scroll_canvas = scroll_canvas
    glasses_window.image_refs = []  # Initialize the image references list
    glasses_window.current_page = 0
    glasses_window.glasses_paths = []

    # Find all image files in the directory
    glasses_paths = []
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    for ext in image_extensions:
        glasses_paths.extend(glob.glob(os.path.join(glasses_dir, ext)))

    glasses_window.glasses_paths = glasses_paths
    glasses_window.total_pages = max(1, (len(glasses_paths) + (number_of_glasses_per_page-1)) // number_of_glasses_per_page)  # Ceiling division by 10

    # Configure pagination buttons
    prev_button.config(command=lambda: load_glasses_page(glasses_window.current_page - 1))
    next_button.config(command=lambda: load_glasses_page(glasses_window.current_page + 1))

    return glasses_window, video_label


def load_glasses_page(page_num):
    """Load a specific page of glasses images (10 per page)"""
    global glasses_window
    global number_of_glasses_per_page

    if not glasses_window:
        print("Window not initialized")
        return

    # Clear previous images and references
    for widget in glasses_window.glasses_frame.winfo_children():
        widget.destroy()
    glasses_window.image_refs = []

    # Update current page
    glasses_window.current_page = page_num
    total_pages = glasses_window.total_pages

    # Update buttons state
    glasses_window.prev_button.config(state="normal" if page_num > 0 else "disabled")
    glasses_window.next_button.config(state="normal" if page_num < total_pages - 1 else "disabled")

    # Update page info
    glasses_window.page_info.config(text=f"Page {page_num + 1} of {total_pages}")

    # Calculate start and end indices for this page
    paths = glasses_window.glasses_paths
    start_idx = page_num * number_of_glasses_per_page
    end_idx = min(start_idx + number_of_glasses_per_page, len(paths))

    if not paths:
        glasses_window.page_info.config(text="No glasses images found")
        return

    # Create buttons for the glasses in this page
    image_refs = []
    for i, path in enumerate(paths[start_idx:end_idx]):
        try:
            # Load the image with PIL
            img_pil = Image.open(path)

            # Resize the image
            img_pil = img_pil.resize((120, 80), Image.LANCZOS)

            # Convert to Tkinter PhotoImage
            img_tk = ImageTk.PhotoImage(img_pil)
            image_refs.append(img_tk)  # Keep reference

            # Create a frame for this glasses option
            item_frame = Frame(glasses_window.glasses_frame, borderwidth=2, relief="groove", padx=5, pady=5)
            item_frame.grid(row=0, column=i, padx=5, pady=5)

            # Add the image
            label = Label(item_frame, image=img_tk)
            label.pack()

            # Add the button
            btn = Button(item_frame, text=os.path.basename(path),
                         command=lambda p=path: select_glasses(p))
            btn.pack()

        except Exception as e:
            print(f"Error loading {path}: {e}")

    # Store references
    glasses_window.image_refs = image_refs


def convert_cv_to_tkinter(cv_img):
    """Convert OpenCV image to Tkinter PhotoImage"""
    # Convert from BGR to RGB color format
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_img = Image.fromarray(rgb_img)

    # Convert to Tkinter PhotoImage
    tk_img = ImageTk.PhotoImage(image=pil_img)
    return tk_img


def main():
    global selected_glasses_path

    # Directory containing glasses images
    glasses_dir = os.path.join(os.getcwd(), r'eyeglass_images')
    # Ensure directory exists
    if not os.path.exists(glasses_dir):
        print(f"Warning: Glasses directory not found at {glasses_dir}")
        print("Please specify a valid directory for glasses images")
        return

    # Create the UI components
    ui, video_label = create_ui(glasses_dir)
    if ui is None:
        print("Failed to create UI")
        return

    # Add a flag to track window state
    window_open = True

    # Handle window close event
    def on_window_close():
        nonlocal window_open
        window_open = False
        glasses_window.destroy()
        print("Window closed by user")

    # Bind the close event
    glasses_window.protocol("WM_DELETE_WINDOW", on_window_close)

    # Load the first page of glasses after UI is displayed
    glasses_window.after(100, lambda: load_glasses_page(0))

    # Default glasses path
    selected_glasses_path = next((path for path in glob.glob(os.path.join(os.getcwd(), r'eyeglass_images', '*'))
                                  if path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))), None)
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return

    facemark = cv2.face.createFacemarkLBF()
    facemark_model = "lbfmodel.yaml"
    if os.path.exists(facemark_model):
        facemark.loadModel(facemark_model)
        print("Facial landmark model loaded successfully")
    else:
        print(f"Warning: Facial landmark model not found at {facemark_model}")
        print("Falling back to basic face detection")
        facemark = None

    # Replace with your ESP32's actual IP address from the ESP32 logs
    esp32_ip = '192.168.0.102'
    stream_url = f"http://{esp32_ip}/mjpeg"

    print(f"Attempting to connect to stream at: {stream_url}")

    # Try to connect to the stream
    try:
        stream = urllib.request.urlopen(stream_url, timeout=10)
        print("Connected to stream successfully")
    except Exception as e:
        print(f"Failed to connect to stream: {e}")
        print("Please check:")
        print("1. The ESP32 IP address is correct")
        print("2. The ESP32 is powered on and connected to WiFi")
        print("3. Your computer and ESP32 are on the same network")
        return

    # Get the multipart boundary string from headers
    headers = stream.info()
    content_type = headers.get('Content-Type', '')
    boundary_match = re.search(r'boundary=(.*?)(?:$|;|\s)', content_type)
    if boundary_match:
        boundary = boundary_match.group(1)
    else:
        boundary = "frame"  # Default boundary used in ESP32 code

    print(f"Using boundary: {boundary}")
    boundary_bytes = f'--{boundary}'.encode()
    content_length_pattern = re.compile(br'Content-Length: (\d+)')

    # Cache for processed glasses images
    glasses_cache = {}

    try:
        buffer = b''
        while True:
            try:
                # Process Tkinter events
                glasses_window.update_idletasks()
                glasses_window.update()

                # Read more data
                chunk = stream.read(1024)
                if not chunk:
                    print("End of stream")
                    break

                buffer += chunk

                # Find the boundary
                boundary_start = buffer.find(boundary_bytes)
                if boundary_start == -1:
                    continue

                # Find the end of headers (double newline)
                headers_end = buffer.find(b'\r\n\r\n', boundary_start)
                if headers_end == -1:
                    continue

                # Extract the headers section
                headers_section = buffer[boundary_start:headers_end]

                # Find Content-Length
                match = content_length_pattern.search(headers_section)
                if not match:
                    # No content length found, discard up to boundary and continue
                    buffer = buffer[boundary_start + len(boundary_bytes):]
                    continue

                # Get the JPEG length
                jpeg_length = int(match.group(1))

                # Check if we have the full JPEG data
                jpeg_start = headers_end + 4  # Skip \r\n\r\n
                if len(buffer) < jpeg_start + jpeg_length:
                    # Not enough data yet, continue reading
                    continue

                # Extract the JPEG data
                jpeg_data = buffer[jpeg_start:jpeg_start + jpeg_length]

                # Update buffer to remove processed data
                buffer = buffer[jpeg_start + jpeg_length:]

                # Process the JPEG data
                frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    # Check if we have a selected glasses image
                    if selected_glasses_path is not None:
                        # Check if we have this glasses image in cache
                        if selected_glasses_path not in glasses_cache:
                            glasses_img = make_white_transparent(selected_glasses_path)
                            if glasses_img is not None:
                                glasses_cache[selected_glasses_path] = glasses_img
                                print(f"Added glasses to cache: {selected_glasses_path}")

                        glasses_img = glasses_cache.get(selected_glasses_path)

                        if glasses_img is not None:
                            # Convert frame to grayscale for face detection
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                            # Detect faces with more lenient parameters
                            faces = face_cascade.detectMultiScale(
                                gray,
                                scaleFactor=1.05,
                                minNeighbors=3,
                                minSize=(30, 30)
                            )

                            # Initialize landmark data
                            face_landmarks = None

                            # Static variable to store last detected face
                            if not hasattr(main, "last_face"):
                                main.last_face = None
                                main.last_landmarks = None
                                main.frames_since_detection = 0

                            if len(faces) > 0:
                                # We found faces, update tracking
                                main.last_face = faces[0]  # Track first face
                                main.frames_since_detection = 0

                                # Try to detect facial landmarks with OpenCV if facemark is available
                                if facemark is not None:
                                    # Convert detected faces to OpenCV format
                                    faces_for_landmarks = []
                                    for (x, y, w, h) in faces:
                                        faces_for_landmarks.append([x, y, w, h])

                                    ok, landmarks = facemark.fit(gray, np.array(faces_for_landmarks))
                                    if ok and len(landmarks) > 0:
                                        face_landmarks = landmarks
                                        main.last_landmarks = face_landmarks

                                        # Add glasses with rotation
                                        frame = overlay_glasses(frame, faces[0], glasses_img, face_landmarks)
                                    else:
                                        # Fallback to original method
                                        frame = overlay_glasses(frame, faces[0], glasses_img)
                                else:
                                    # Fallback to original method
                                    frame = overlay_glasses(frame, faces[0], glasses_img)

                            elif main.last_face is not None and main.frames_since_detection < 30:
                                # No face found but use last position (for up to 30 frames)
                                frame = overlay_glasses(frame, main.last_face, glasses_img, main.last_landmarks)
                                main.frames_since_detection += 1

                    tk_frame = convert_cv_to_tkinter(frame)
                    video_label.configure(image=tk_frame)
                    video_label.image = tk_frame  # Keep a reference

                    # Process Tkinter events
                    glasses_window.update()
                else:
                    print("Failed to decode image")
            except Exception as e:
                if not window_open:
                    return
                print(f"Error processing frame: {e}")
                time.sleep(1)
                try:
                    # Keep UI responsive even if there are frame errors
                    glasses_window.update_idletasks()
                    glasses_window.update()
                except:
                    pass

    except KeyboardInterrupt:
        print("Stream viewing stopped by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        try:
            cv2.destroyAllWindows()
            if glasses_window:
                glasses_window.destroy()
        except:
            pass
        print("Stream viewer closed")


if __name__ == "__main__":
    main()
