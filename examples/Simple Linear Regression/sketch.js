//Creating the dataset
let size = 100;
let X_coors = [];
let Y_coors = [];

//Canvas parameters
const width = 800;
const height = 600;

//Variables for regression
let m, b, tfys;

//Neural Net Hyper Parameters
let learning_rate = 0.01;
let optimizer = tf.train.adam(learning_rate);

//Function to predict
function predict(xs){

  const tfxs = tf.tensor1d(xs);
  const ys = tfxs.mul(m).add(c);

  return ys;
}

//Function to predict the loss
function loss(pred, labels){
  return pred.sub(labels).square().mean(); 
}

function setup(){
   
   createCanvas(width, height);
   background(0);

   m = tf.variable(tf.scalar(Math.random()));
   c = tf.variable(tf.scalar(Math.random()));


}

function draw(){

  //Clearing screen
  background(0);
 
  //Drawing the points
  for (let i = 0; i < X_coors.length; ++i)
  {
  	fill(255);
  	ellipse(X_coors[i] * width, Y_coors[i] * height , 10 , 10);

  	//Drawing the line
    xs = [0, 1];
    ys = predict(xs).dataSync();
    stroke(255);
    strokeWeight(2);	
    line(xs[0] * width, ys[0] * height, xs[1] * width, ys[1] * height);

  }

 	 
  //Minimizing if there are points in the array
  if (X_coors.length > 0){
  	tfys = tf.tensor1d(Y_coors);
    optimizer.minimize(() => loss(predict(X_coors), tfys));
  }

 } 

function mousePressed(){
  X_coors.push(mouseX / width);
  Y_coors.push(mouseY / height);
}