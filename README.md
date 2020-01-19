# TrafficSignRecognition

Traffic signs are an integral part of our road infrastructure. Traffic signs are road facilites that convey, guide, restrict, warn or instruct information using words or symbols. Traffic sign recognition plays an important role in expert systems such as traffic assistance driving systems and automatic driving systems.

You all must have heard about the self-driving cars in which the passengers can fully depend on the car for travelling. But to achieve level 5 autonomous, it is necessary for vehicles to understand and follow traffic rules. With the development of automative intelligent technology famous car companies, such as BMW, Mercedes-Benz, Audi are investing on ADAS(Advanced Driver Assistance System) research. ADAS includes TRS(Traffic Recognition System) along with some other advanced traffic assistance systems.

# what is traffic sign recognition ?

Traffic sign recognition is a process of classifying a traffic sign into different categories. Traffic-sign recognition (TSR) is a technology by which a vehicle is able to recognize the traffic signs put on the road e.g. "speed limit" or "children" or "turn ahead". This is part of the features collectively called ADAS. The technology is being developed by a variety of automotive suppliers. It uses image processing techniques to detect the traffic signs. The detection methods can be generally divided into color based, shape based and learning based methods.

# About Dataset

The Dataset contains about 39210 image links of different traffic signs in Train.csv. The given images can be classified into 43 different classes. The given Dataset is little bit imbalanced some of the classes have many images while some classes have few images. The size of the dataset is 300MB. The dataset has a train folder which contains images inside each class and a test folder for testing the developed model.

There are 43 Different categories that are present in the dataset like no entry, speed limit, children crossing, etc. This web app can detect traffic signs in the uploaded image and can classify into different categories.

Speed limit (20km/h)
Speed limit (30km/h)
Speed limit (50km/h)
Speed limit (60km/h)
Speed limit (70km/h)
Speed limit (80km/h)
End of Speed limit (80km/h)
Speed limit (100km/h)
Speed limit (100km/h)
Speed limit (120km/h)
No Passing
No Passing veh over 3.5 tons
Right-of-way at intersection
Priority Road
Yield
Stop
No Vehicles
Veh > 3.5 tons Prohibited
No entry
General Caution
Dangerous Curve Left
Dangerous Curve Right
Double Curve
Bumpy Road
Slippery Road
Road Narrows On The Right
Road Work
Traffic Signals
Pedestrians
Children Crossing
Bicycles Crossing
Beware Of ice/Snow
Wild Animals Crossing
End Speed + Passing Limits
Turn Right Ahead
Turn Left Ahead
Ahead Only
Go Straight Or Right
Go Straight or Left
Keep Right
Keep Left
Roundabout mandatory
End Of No Passing
End No Passing Veh > 3.5 tons
Dataset link :Traffic Signs Dataset Kaggle

# About Model

A CNN(Convolutional Neural Network) model is trained on the dataset, as CNN is best suited for image classification related tasks.

Following is the architecture of the CNN model :

2 Conv2D layers with filter = 32, kernel size = (5,5) and activation function as relu.
Maxpool2D layer with pool size = (2,2)
A Dropout layer with rate = 0.25
2 Conv2D layers with filter = 64, kernel size = (3,3) and activation function as relu.
Maxpool2D layer with pool size = (2,2)
A Dropout layer with rate = 0.25
Flatten layer to squeeze the layers into one dimension.
A Dense fully connected layer with 256 nodes and relu as activation function.
Dropout layer with rate as 0.5
At last a Dense layer with 43 nodes and Sigmoid as activation function.

