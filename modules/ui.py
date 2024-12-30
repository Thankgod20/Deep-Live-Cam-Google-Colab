import sys
import os
import webbrowser
import customtkinter as ctk
import tkinter as tk
from typing import Callable, Tuple
import cv2
from PIL import Image, ImageOps
from typing import Any, List, Callable
from types import ModuleType
import modules.globals
import modules.metadata
from modules.face_analyser import get_one_face
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import is_image, is_video, resolve_relative_path
import queue
import threading
import time
ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 800

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None

img_ft, vid_ft = modules.globals.file_types
frames_array =[]

def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT
def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(f'{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}')
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    # Labels for source and target
    source_label = ctk.CTkLabel(root, text="Source")
    source_label.place(relx=0.05, rely=0.05, relwidth=0.4, relheight=0.1)

    target_label = ctk.CTkLabel(root, text="Target")
    target_label.place(relx=0.55, rely=0.05, relwidth=0.4, relheight=0.1)

    # Buttons for selecting source and target
    select_face_button = ctk.CTkButton(root, text='Select a face', cursor='hand2', command=lambda: select_source_path())
    select_face_button.place(relx=0.05, rely=0.15, relwidth=0.4, relheight=0.08)

    select_target_button = ctk.CTkButton(root, text='Select a target', cursor='hand2', command=lambda: select_target_path())
    select_target_button.place(relx=0.55, rely=0.15, relwidth=0.4, relheight=0.08)

    
    # Switches for settings
    settings_frame = ctk.CTkFrame(root)
    settings_frame.place(relx=0.05, rely=0.3, relwidth=0.9, relheight=0.3)

    ctk.CTkLabel(settings_frame, text="Settings").grid(row=1, column=1, padx=10, pady=10, sticky='w')#.grid(anchor='w', pady=5)
    
    switches = [
        ("Keep FPS", modules.globals.keep_fps, lambda value: setattr(modules.globals, 'keep_fps', value)),
        ("Keep Frames", modules.globals.keep_frames, lambda value: setattr(modules.globals, 'keep_frames', value)),
        ("Face Enhancer", modules.globals.fp_ui['face_enhancer'], lambda value: update_tumbler('face_enhancer', value)),
        ("Keep Audio", modules.globals.keep_audio, lambda value: setattr(modules.globals, 'keep_audio', value)),
        ("Many Faces", modules.globals.many_faces, lambda value: setattr(modules.globals, 'many_faces', value)),
        ("NSFW", modules.globals.nsfw, lambda value: setattr(modules.globals, 'nsfw', value)),
        ("Remote Processor", modules.globals.fp_ui['remote_processor'], lambda value: update_tumbler('remote_processor', value)),
    ]

    for idx_r,(text, var, cmd) in enumerate(switches):
        val = ctk.BooleanVar(value=var)
       # ctk.CTkSwitch(settings_frame, text=text, variable=val, command=lambda v=val: cmd(v.get())).pack(anchor='w', pady=2)
        row, col = divmod(idx_r, 3)
        val = ctk.BooleanVar(value=var)
        ctk.CTkSwitch(settings_frame, text=text, variable=val, command=lambda v=val: cmd(v.get())).grid(row=row, column=col, padx=10, pady=10, sticky='w')
        # Create the streaming frame
        streaming_frame = ctk.CTkFrame(root)
        streaming_frame.place(relx=0.05, rely=0.65, relwidth=0.9, relheight=0.15)

        # Add the "Streaming Settings" label with grid
        streaming_label = ctk.CTkLabel(streaming_frame, text="Streaming Settings")
        streaming_label.grid(row=0, column=0, columnspan=4, sticky='w', pady=5)

        # Define inputs and their arrangement
        inputs = [("GoF", 'gof'), ("Bitrate", 'bitrate'), ("Maxrate", 'maxrate'), ("Bufsize", 'bufsize')]
        default_values = {
            'gof': '15',  # Default value for GoF
            'bitrate': '800k',  # Default value for Bitrate
            'maxrate': '800k',  # Default value for Maxrate
            'bufsize': '800k',  # Default value for Bufsize
        }
        def get_textbox_width_percentage(parent_frame, percentage):
            # Get the width of the parent frame
            frame_width = parent_frame.winfo_width()
            # Calculate the width as a number of characters (approximate)
            return int((frame_width * percentage) / 100)
        
        def set_textbox_width_percentage():
            for idx, (label_text, attr) in enumerate(inputs):
                ctk.CTkLabel(streaming_frame, text=f"{label_text}:").grid(row=1, column=idx, padx=5, sticky='w')
                # Calculate the width in percentage (e.g., 50% of the parent frame's width)
                textbox_width = get_textbox_width_percentage(streaming_frame, 22)  # 50% width
        
                text_box = ctk.CTkTextbox(streaming_frame, height=20,width=textbox_width)
                # Set the default value
                text_box.insert("1.0", default_values.get(attr, ""))  # Insert default value
                text_box.bind("<KeyRelease>", lambda e, a=attr, tb=text_box: setattr(modules.globals, a, tb.get("1.0", tk.END).strip()))
                text_box.grid(row=2, column=idx, padx=5, sticky='w')

        root.after(100, set_textbox_width_percentage) 

    # Textboxes for addresses
    address_frame = ctk.CTkFrame(root)
    address_frame.place(relx=0.05, rely=0.82, relwidth=0.9, relheight=0.1)

    ctk.CTkLabel(address_frame, text="In:").grid(row=0, column=0, padx=5)
    text_box_addr_in = ctk.CTkTextbox(address_frame, height=20)
    text_box_addr_in.bind("<KeyRelease>", lambda e, a="pull_addr", tb=text_box_addr_in: setattr(modules.globals, a, tb.get("1.0", tk.END).strip()))
    text_box_addr_in.grid(row=0, column=1, padx=5)

    ctk.CTkLabel(address_frame, text="Out S:").grid(row=0, column=2, padx=5)
    text_box_addr_out = ctk.CTkTextbox(address_frame, height=20)
    text_box_addr_out.bind("<KeyRelease>", lambda e, a="push_addr", tb=text_box_addr_out: setattr(modules.globals, a, tb.get("1.0", tk.END).strip()))
    text_box_addr_out.grid(row=0, column=3, padx=5)

    ctk.CTkLabel(address_frame, text="Out T:").grid(row=0, column=4, padx=5)
    text_box_addr_out_t = ctk.CTkTextbox(address_frame, height=20)
    text_box_addr_out_t.bind("<KeyRelease>", lambda e, a="push_addr_two", tb=text_box_addr_out_t: setattr(modules.globals, a, tb.get("1.0", tk.END).strip()))
    text_box_addr_out_t.grid(row=0, column=5, padx=5)

    # Action buttons
    button_frame = ctk.CTkFrame(root)
    button_frame.place(relx=0.05, rely=0.93, relwidth=0.9, relheight=0.05)

    ctk.CTkButton(button_frame, text='Start', cursor='hand2', command=lambda: select_output_path(start)).pack(side='left', expand=True, padx=5)
    ctk.CTkButton(button_frame, text='Destroy', cursor='hand2', command=lambda: destroy()).pack(side='left', expand=True, padx=5)
    ctk.CTkButton(button_frame, text='Preview', cursor='hand2', command=lambda: toggle_preview()).pack(side='left', expand=True, padx=5)
    ctk.CTkButton(button_frame, text='Live', cursor='hand2', command=lambda: webcam_preview()).pack(side='left', expand=True, padx=5)
    
    # Status label
    status_label = ctk.CTkLabel(root, text=None, justify='center')
    status_label.place(relx=0.05, rely=0.98, relwidth=0.9)

    return root

''' 
def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(f'{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}')
    root.configure()
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    source_label = ctk.CTkLabel(root, text=None)
    source_label.place(relx=0.1, rely=0.0, relwidth=0.25, relheight=0.2)

    target_label = ctk.CTkLabel(root, text=None)
    target_label.place(relx=0.6, rely=0.0, relwidth=0.25, relheight=0.2)

    select_face_button = ctk.CTkButton(root, text='Select a face', cursor='hand2', command=lambda: select_source_path())
    select_face_button.place(relx=0.1, rely=0.15, relwidth=0.3, relheight=0.1)

    select_target_button = ctk.CTkButton(root, text='Select a target', cursor='hand2', command=lambda: select_target_path())
    select_target_button.place(relx=0.6, rely=0.15, relwidth=0.3, relheight=0.1)

    keep_fps_value = ctk.BooleanVar(value=modules.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(root, text='Keep fps', variable=keep_fps_value, cursor='hand2', command=lambda: setattr(modules.globals, 'keep_fps', not modules.globals.keep_fps))
    keep_fps_checkbox.place(relx=0.1, rely=0.5)

    keep_frames_value = ctk.BooleanVar(value=modules.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(root, text='Keep frames', variable=keep_frames_value, cursor='hand2', command=lambda: setattr(modules.globals, 'keep_frames', keep_frames_value.get()))
    keep_frames_switch.place(relx=0.1, rely=0.55)

    # for FRAME PROCESSOR ENHANCER tumbler:
    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui['face_enhancer'])
    enhancer_switch = ctk.CTkSwitch(root, text='Face Enhancer', variable=enhancer_value, cursor='hand2', command=lambda: update_tumbler('face_enhancer',enhancer_value.get()))
    enhancer_switch.place(relx=0.1, rely=0.6)

    remote_process_value = ctk.BooleanVar(value=modules.globals.fp_ui['remote_processor'])
    remote_process_switch = ctk.CTkSwitch(root, text='Remote Processor', variable=remote_process_value, cursor='hand2', command=lambda: update_tumbler('remote_processor',remote_process_value.get()))
    remote_process_switch.place(relx=0.1, rely=0.65)
   
    def on_text_change(event=None):
        setattr(modules.globals, 'pull_addr', text_box_addr_in.get("1.0", tk.END).strip())
    
    def on_text_change_out(event=None):
        setattr(modules.globals, 'push_addr', text_box_addr_out.get("1.0", tk.END).strip())
    def on_text_change_out_two(event=None):
        setattr(modules.globals, 'push_addr_two', text_box_addr_out_t.get("1.0", tk.END).strip())

    #streaming settings
    def on_text_change_gof(event=None):
        setattr(modules.globals, 'gof', text_box_addr_in.get("1.0", tk.END).strip())
    def on_text_change_bitrate(event=None):
        setattr(modules.globals, 'bitrate', text_box_addr_in.get("1.0", tk.END).strip())
    def on_text_change_maxrate(event=None):
        setattr(modules.globals, 'maxrate', text_box_addr_in.get("1.0", tk.END).strip())
    def on_text_change_bufsize(event=None):
        setattr(modules.globals, 'bufsize', text_box_addr_in.get("1.0", tk.END).strip())
    
    keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(root, text='Keep audio', variable=keep_audio_value, cursor='hand2', command=lambda: setattr(modules.globals, 'keep_audio', keep_audio_value.get()))
    keep_audio_switch.place(relx=0.6, rely=0.5)

    many_faces_value = ctk.BooleanVar(value=modules.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(root, text='Many faces', variable=many_faces_value, cursor='hand2', command=lambda: setattr(modules.globals, 'many_faces', many_faces_value.get()))
    many_faces_switch.place(relx=0.6, rely=0.55)

    nsfw_value = ctk.BooleanVar(value=modules.globals.nsfw)
    nsfw_switch = ctk.CTkSwitch(root, text='NSFW', variable=nsfw_value, cursor='hand2', command=lambda: setattr(modules.globals, 'nsfw', nsfw_value.get()))
    nsfw_switch.place(relx=0.6, rely=0.6)
    ''' #Label a text box 
    #label = ctk.CTkLabel(root, text="In:")
    #label.place(relx=0.1, rely=0.72, anchor=tk.E)
    #Label a text box
    #label = ctk.CTkLabel(root, text="OutS:")
    #label.place(relx=0.6, rely=0.72, anchor=tk.E)
'''
     # Streaming setting a text box gof
    text_box_gof = ctk.CTkTextbox(root, width=20, height=20)
    text_box_gof.place(relx=0.1, rely=0.6)
    text_box_gof.bind("<KeyRelease>", on_text_change_gof)
    # Streaming setting a text box bitrate
    text_box_bitrate = ctk.CTkTextbox(root, width=20, height=20)
    text_box_bitrate.place(relx=0.2, rely=0.6)
    text_box_bitrate.bind("<KeyRelease>", on_text_change_bitrate)

    # Streaming setting a text box bitrate
    text_box_maxrate = ctk.CTkTextbox(root, width=20, height=20)
    text_box_maxrate.place(relx=0.3, rely=0.6)
    text_box_maxrate.bind("<KeyRelease>", on_text_change_maxrate)

    # Streaming setting a text box bitrate
    text_box_bufsize = ctk.CTkTextbox(root, width=20, height=20)
    text_box_bufsize.place(relx=0.4, rely=0.6)
    text_box_bufsize.bind("<KeyRelease>", on_text_change_bufsize)

    #Label a text box
    label = ctk.CTkLabel(root, text="In:")
    label.place(relx=0.1, rely=0.72, anchor=tk.E)
    #Label a text box
    label = ctk.CTkLabel(root, text="OutS:")
    label.place(relx=0.6, rely=0.72, anchor=tk.E)
    # Create a text box
    text_box_addr_in = ctk.CTkTextbox(root, width=200, height=10)
    text_box_addr_in.place(relx=0.1, rely=0.75)
    text_box_addr_in.bind("<KeyRelease>", on_text_change)

    # Create a text box
    text_box_addr_out = ctk.CTkTextbox(root, width=100, height=10)
    text_box_addr_out.place(relx=0.6, rely=0.75)
    text_box_addr_out.bind("<KeyRelease>", on_text_change_out)

    # Create a text box
    text_box_addr_out_t = ctk.CTkTextbox(root, width=100, height=10)
    text_box_addr_out_t.place(relx=0.8, rely=0.75)
    text_box_addr_out_t.bind("<KeyRelease>", on_text_change_out_two)

    start_button = ctk.CTkButton(root, text='Start', cursor='hand2', command=lambda: select_output_path(start))
    start_button.place(relx=0.15, rely=0.9, relwidth=0.2, relheight=0.05)

    stop_button = ctk.CTkButton(root, text='Destroy', cursor='hand2', command=lambda: destroy())
    stop_button.place(relx=0.4, rely=0.9, relwidth=0.2, relheight=0.05)

    preview_button = ctk.CTkButton(root, text='Preview', cursor='hand2', command=lambda: toggle_preview())
    preview_button.place(relx=0.65, rely=0.9, relwidth=0.2, relheight=0.05)

    live_button = ctk.CTkButton(root, text='Live', cursor='hand2', command=lambda: webcam_preview())
    live_button.place(relx=0.40, rely=1.5, relwidth=0.2, relheight=0.05)

    status_label = ctk.CTkLabel(root, text=None, justify='center')
    status_label.place(relx=0.1, rely=0.95, relwidth=0.8)

    donate_label = ctk.CTkLabel(root, text='Deep Live Cam', justify='center', cursor='hand2')
    donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)
    donate_label.configure(text_color=ctk.ThemeManager.theme.get('URL').get('text_color'))
    donate_label.bind('<Button>', lambda event: webbrowser.open('https://paypal.me/hacksider'))

    return root
'''

def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('Preview')
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='both', expand=True)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(title='select an source image', initialdir=RECENT_DIRECTORY_SOURCE, filetypes=[img_ft])
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(title='select an target image or video', initialdir=RECENT_DIRECTORY_TARGET, filetypes=[img_ft, vid_ft])
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save image output file', filetypes=[img_ft], defaultextension='.png', initialfile='output.png', initialdir=RECENT_DIRECTORY_OUTPUT)
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save video output file', filetypes=[vid_ft], defaultextension='.mp4', initialfile='output.mp4', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        start()


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()
        PREVIEW.deiconify()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill='x')
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        remote_process=False
        if modules.globals.nsfw == False:
            from modules.predicter import predict_frame
            if predict_frame(temp_frame):
                quit()
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            if 'remote_processor' in modules.globals.frame_processors :
                remote_process = True
                if frame_processor.__name__ =="modules.processors.frame.remote_processor":
                    print('------- Remote Process ----------')
                    source_data = cv2.imread(modules.globals.source_path)
                    if not frame_processor.pre_check():
                        print("No Input and Output Address")
                        sys.exit()
                    temp_frame=frame_processor.process_frame(source_data,temp_frame)
                    if temp_frame.ndim == 4:
                        temp_frame = temp_frame[0]

            if not remote_process:
                temp_frame = frame_processor.process_frame(
                    get_one_face(cv2.imread(modules.globals.source_path)),
                    temp_frame
                )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
terminate_flag = threading.Event() 
def send_streams(cap:cv2.VideoCapture,input_queue:queue.Queue,remote_modules:List[ModuleType],stream_out:Any):
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cutting cam...")
                break
            temp_frame = frame.copy() 
            #input_queue.put(temp_frame)
            remote_modules[1].write_to_stdin(temp_frame,input_queue,stream_out)
    finally:
        cap.release()
        stream_out.terminate()
        print("Stream out terminated.")
def receive_frames(output_queue:queue.Queue,remote_modules:List[ModuleType],stream_in:Any):
    try:
        while True:
            remote_modules[1].read_from_stdout(stream_in,output_queue)
            while not output_queue.empty():
                item = output_queue.get()
                frames_array.append(item)
    finally:
        stream_in.terminate()  
        print("Stream in terminated.") 

def webcam_preview():
    if modules.globals.source_path is None:
        # No image selected
        return

    global preview_label, PREVIEW

    cap = cv2.VideoCapture(0)  # Use index for the webcam (adjust the index accordingly if necessary)    
    cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # Set the width of the resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)  # Set the height of the resolution
    cap.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate of the webcam
    PREVIEW_MAX_WIDTH = 960
    PREVIEW_MAX_HEIGHT = 540

    preview_label.configure(image=None)  # Reset the preview image before startup

    PREVIEW.deiconify()  # Open preview window

    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)

    source_image = None  # Initialize variable for the selected face image
    remote_process=False # By default remote process is set to disabled
    stream_out = None   # Both veriable stores the subprocess runned by ffmpeg
    stream_in = None

    if 'remote_processor' in modules.globals.frame_processors :
        print(f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        remote_modules = get_frame_processors_modules(['remote_processor'])
        source_data = cv2.imread(modules.globals.source_path)
        remote_modules[1].send_source_frame(source_data)
        #start ffmpeg stream out subprocess
        stream_out = remote_modules[1].send_streams(cap)
        #start ffmpeg stream In subprocess
        stream_in = remote_modules[1].recieve_streams(cap)
        remote_process = True
        input_queue = queue.Queue()
        output_queue = queue.Queue()
        
        thread_stream = threading.Thread(target=send_streams,args=(cap,input_queue,remote_modules,stream_out,))
        thread_stream.start()

        thread_stream_in = threading.Thread(target=receive_frames,args=(output_queue,remote_modules,stream_in,))
        thread_stream_in.start()
        global frames_array
        try:
            while True:
                if len(frames_array) > 0:
                    frame = frames_array[-1]  # Display the latest frame
                    cv2.imshow("Processed Stream", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord("x"):
                        print("Exiting Program...")
                        terminate_flag.set()
                        break
                    
                    if len(frames_array) > 1:
                        frames_array.pop(0)
        finally:
            terminate_flag.set()  # Signal threads to stop
            print("Terminating FFmpeg subprocesses...")
            stream_out.terminate()
            stream_out.kill()  # Ensure termination
            stream_in.terminate()
            stream_in.kill()
            print("Terminating threads...")
            thread_stream.join()
            thread_stream_in.join()

            print("Releasing resources...")
            cap.release()  # Release webcam
            input_queue.queue.clear()
            output_queue.queue.clear()
            PREVIEW.withdraw()  # Close preview window when loop is finished
            
    else:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Select and save face image only once
                if source_image is None and modules.globals.source_path:
                    source_image = get_one_face(cv2.imread(modules.globals.source_path))

                temp_frame = frame.copy()  #Create a copy of the frame

                for frame_processor in frame_processors:
                    if not remote_process:
                        temp_frame = frame_processor.process_frame(source_image, temp_frame)
                
                image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)  # Convert the image to RGB format to display it with Tkinter
                image = Image.fromarray(image)
                image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
                image = ctk.CTkImage(image, size=image.size)
                preview_label.configure(image=image)
                ROOT.update()
                  
                if PREVIEW.state() == 'withdrawn':
                    break
        finally:
            cap.release()
            PREVIEW.withdraw()  # Close preview window when loop is finished
