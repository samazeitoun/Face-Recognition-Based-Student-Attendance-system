import cv2, os, numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def capture_images(name, count=5):
    os.makedirs(f"students_photos/{name}", exist_ok=True)
    cap = cv2.VideoCapture(0)
    print(f"Capturing for {name}: press 'c' to snap, 'q' to quit.")
    n=0
    while n<count:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow(name, frame)
        k=cv2.waitKey(1)&0xFF
        if k==ord('c'):
            cv2.imwrite(f"students_photos/{name}/{name}_{n}.jpg", frame)
            print(f"Captured {n+1}/{count}")
            n+=1
        elif k==ord('q'): break
    cap.release(); cv2.destroyAllWindows()

def load_data(folder="students_photos"):
    X,y=[],[]
    for name in os.listdir(folder):
        path=os.path.join(folder,name)
        if os.path.isdir(path):
            for img in os.listdir(path):
                im=cv2.imread(os.path.join(path,img),0)
                if im is not None:
                    im=cv2.resize(im,(50,50))
                    X.append(im.flatten())
                    y.append(name)
    return np.array(X), np.array(y)

def train():
    X,y=load_data()
    if len(X)==0:
        print("No images!")
        return None
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train,y_train)
    print(f"Accuracy: {model.score(X_test,y_test)*100:.2f}%")
    return model
def recognize(model):
    fc=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    cap=cv2.VideoCapture(0)
    print("Press 's' to snap group, 'q' to quit.")
    all_students = os.listdir("students_photos")  # كل الطلاب المسجلين
    while True:
        ret,frame=cap.read()
        if not ret: break
        cv2.imshow("Recognition", frame)
        k=cv2.waitKey(1)&0xFF
        if k==ord('s'):
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=fc.detectMultiScale(gray,1.1,5)
            if len(faces)==0:
                print("No faces found.")
                continue
            present = []
            for (x,y,w,h) in faces:
                face=gray[y:y+h,x:x+w]
                face=cv2.resize(face,(50,50)).flatten().reshape(1,-1)
                pred=model.predict(face)[0]
                present.append(pred)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,pred,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            present = list(set(present))
            absent = list(set(all_students) - set(present))
            print("Present:", ", ".join(present))
            print("Absent:", ", ".join(absent) if absent else "None")
            cv2.imshow("Recognition", frame)
            cv2.waitKey(3000)
        elif k==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


model=None
while True:
    print("\n1: Register\n2: Train\n3: Recognize\n4: Exit")
    c=input("Choose: ")
    if c=="1": capture_images(input("Name: "))
    elif c=="2": model=train()
    elif c=="3":
        if model: recognize(model)
        else: print("Train model first!")
    elif c=="4": break
    else: print("Invalid.")
