# RouteAR

## Techincal Modules:

- 3D representation of crags + routes
- Position + orientation of phone realtive to crag.
- accurate AR 

### Available data

- GPS - which includes elevation. 
- IMU - gets us direction and orientation
- camera - hopefully we can the internal camera parameters for each phone...

## Labeling
Will need to label the routes by hand to start to ensure accuracy. Get a program going for people to label their own crags.

## Methods


### Simple solution 
We have GPS-located routes and and 3D model of routes around the world. Using the GPS and orientation of your phone we can kind of point out where each route is. This is basically how the star apps work, I believe. But accuracy is not as important on those apps. 

### SFM 
Use SFM to get a 3D model of crag structure, then use new frame registration to understand where the amera is looking. Possible issues with real time. Especially if we need to use bundle adjustment. 

### Features
Use different features to define the "structure" of a route or crag. If those features can then be found reliably again, we can do some geometry to get a relationship of the camera to the original dataset and then do that with the representation of the route. 




