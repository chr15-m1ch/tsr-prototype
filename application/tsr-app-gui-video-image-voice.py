# better version, where model is loaded one time
# features: gui, realtime, video, image, voice notification

# imports

# gui imports
from tkinter import *
from PIL import ImageTk,Image
import runpy
from tkinter import filedialog


# declare my global variables
root = Tk()

# function to trigger realtime btn
def realTime():
    print("real time")
    # runpy.run_path("skipframes/realtime.py")
    
# function to trigger upload video btn
def uploadVideo():
    print("upload video")
    # runpy.run_path("classify_subset_video.py")
    # runpy.run_path("skipframes/video.py")

def uploadImage():
    print("upload image")
    # runpy.run_path("classify_subset_image.py")


def center(frm): #center a frame
    # size of the app
    app_width = 650
    app_height = 550
    # size of screen
    screen_width = frm.winfo_screenwidth()
    screen_height = frm.winfo_screenheight()
    x = (screen_width / 2) - (app_width / 2)
    y = (screen_height / 2) - (app_height / 2)
    # position window in center of screen
    frm.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')
    frm.mainloop()


def main():
    # first we build our user interface

    root.title('Traffic Sign Recognition using Deep Learning')
    # icon of the app
    root.iconbitmap('app_images/traffic-512.ico')
    # add image to window
    my_img = ImageTk.PhotoImage(Image.open("app_images/signs.jpg"))
    my_label = Label(image=my_img)
    my_label.pack()
    # adding btns to the window
    btnFrame = LabelFrame(root, text="Actions", padx=100, pady=10)
    btnFrame.pack(padx=10, pady=10)
    btn1 = Button(btnFrame, text="Real Time", command=realTime)
    btn1.grid(row=0, column=0, padx=5)
    btn2 = Button(btnFrame, text="Upload Video", command=uploadVideo)
    btn2.grid(row=0, column=1, padx=5)
    btn3 = Button(btnFrame, text="Upload Image", command=uploadImage)
    btn3.grid(row=0, column=2, padx=5)
    btn4 = Button(btnFrame, text="Close", command=root.quit)
    btn4.grid(row=0, column=3, padx=5)
    # center the frame
    center(root)

    


if __name__ == '__main__':
    main()
