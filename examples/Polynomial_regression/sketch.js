//Creating the dataset
let size = 100;
let X_coors = [];
let Y_coors = [];

//Canvas parameters
const width = 800;
const height = 600;

//Variables for regression
let a, b, c, tfys;
let xs = [];
let total = 200;

//Neural Net Hyper Parameters
let learning_rate = 0.01;
let optimizer = tf.train.adam(learning_rate);

//Function to predict
function predict(xs){

  const tfxs = tf.tensor1d(xs);
  const quad_term = tf.square(tfxs).mul(a);
  const lin_term = tfxs.mul(b);
  const ys = quad_term.add(lin_term).add(c);

  return ys;
}

//Function to predict the loss
function loss(pred, labels){
  return pred.sub(labels).square().mean(); 
}

function setup(){
   
   createCanvas(width, height);
   background(0);

   a = tf.variable(tf.scalar(Math.random()));
   b = tf.variable(tf.scalar(Math.random()));
   c = tf.variable(tf.scalar(Math.random()));

   for (let i = 0; i < total; i++)  
   {
     xs.push(i / total);
   }

}

function draw(){

  //Clearing screen
  background(0);
 
  //Drawing the points
  for (let i = 0; i < X_coors.length; ++i)
  {
  	fill(255);
  	ellipse(X_coors[i] * width, Y_coors[i] * height , 10 , 10);
  }

  //Drawing the prediction
  ys = predict(xs).dataSync();
  
  for (let i = 0; i < ys.length; ++i)
  {
    fill(255, 0 , 120);
    ellipse(xs[i] * width, ys[i] * height , 5 , 5);  
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
