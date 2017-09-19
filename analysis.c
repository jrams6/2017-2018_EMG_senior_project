
#include "Header.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
//#include <time.h>

//General Defines
#define ARRAYSIZE 4										//
#define AdcClk 30e6								    //ADC clock frequency
#define cycles 480										//# of ADC cycles before sample
#define MaxTime 275										//max time threshold
#define MinTime 50										//min time threshold
#define SamplingTime .000016                 //(1/(float)AdcClk/prescaler)*cycles   //prescaler = 6
#define MINLENGTH 700                    //(MinTime*.001)/SamplingTime          //Min # of samples
#define MAXLENGTH 15000
//const int MAXLENGTH = (int)((MaxTime*.001)/SamplingTime);           //max # of samples

#define CAL_NUM 5										  //# of calibration motions necessary - 1
#define MOTIONS 4										  //# of Motions - 1
#define THRES 400	                                    //horizontal level threshold

//Neural Network
//#define MAX_SLOPE_CHANGES 5500                            //Max number of slope changes
//#define MAX_WAVE_COMPLEXITY 35000                        //Max Wave complexity
#define INPUT_LAYER 2
#define HIDDEN_LAYER 3
#define OUTPUT_LAYER 4
#define A INPUT_LAYER + HIDDEN_LAYER + OUTPUT_LAYER
#define WEIGHTS_NUM (INPUT_LAYER*HIDDEN_LAYER)+(OUTPUT_LAYER*HIDDEN_LAYER)
#define B A-INPUT_LAYER

#define FEAT 2

//constant for Network
const uint8_t nodes[] = {2, 3, 4};

//Matrix as 1d array definitions
static float a[A];
static float z[B];
static float weights[WEIGHTS_NUM];
static float err[B];
static float bias[B];

//val segmenting variables
static uint8_t flag = 0;										//flag indicating tracking of signal
static uint8_t wait = 0;										//wait flag indicating the signal is longer than MAXLENGTH
static uint8_t analyze = 0;									//Analyze flag indicating the end of signal tracking and need to analyze
//static uint32_t last_data = 0;								//log data during tracking

//Features
static uint32_t count = 0; 									//Length of signal
static float features[FEAT] = {0, 0};
static float max_values[FEAT] = {0, 0};
static float min_values[FEAT] = {0, 0};
float mean_ov = 0;


//Function prototypes
void Val();
void SegAnalysis(uint32_t val, uint8_t end);
void Operational();
void calibrate();

//Network
float softmax(float i);
float SigFunc(float i);
void Input_init_cal(float max_values[], float min_values[], float input[], uint8_t motion, uint8_t cal);
void Input_init_Op(float max_values[], float min_values[], float input[]);
void backpropagation(uint8_t , float [], float, uint8_t);
void feedforward(uint8_t Layers);
void Network_Shutdown();
void Network_Setup();
//Network Matrix indexing
uint8_t A_indexing(uint8_t dimension, uint8_t node);
uint8_t B_indexing(uint8_t dimension, uint8_t node);
uint8_t weight_indexing(uint8_t dimension, uint8_t lnode, uint8_t node);
inline uint8_t O_indexing(uint8_t i, uint8_t q);
inline uint8_t I_indexing(uint8_t motion, uint8_t cal, uint8_t index);


/**************************ANALYSIS.C*****************************/
/*
*** TITLE: Calibrate Function
*** FUNCTION: Takes user data to inialize the system
***	INPUTS: NONE
*** OUTPUTS: NONE
*/

void calibrate(){
	//Variables
  uint8_t i = 0;                            //for loop index
  uint8_t q = 0;                            //..
  float LR = .1;									//Learning Rate
  double cost = 0;
  double last_cost = 0;
  float max[FEAT] = {0, 0};
  float min[FEAT] = {0, 0};

	//Calibration Variables								// temp wave complexity
  uint8_t testNum = 0;                      //current number of calibration motions completed

  //Network Variables --------------------------------------
  //number samples for each motion for training
  uint8_t n[MOTIONS] = {5, 5, 5, 5};
  //number of layers in the network
  uint8_t Layers = (sizeof(nodes) / sizeof(nodes[0]));
  //Actual outputs for training
  float y[MOTIONS*nodes[Layers-1]];                            //# of MOTIONS ^ 2
  //feature input for all motions and calibration sets
  float input[MOTIONS*(CAL_NUM*(FEAT))];
 

  //Output array inialization
  for (i = 0; i < MOTIONS; i++){
    for (q = 0; q < nodes[Layers-1]; q++){
      if (q == i){
        y[O_indexing(i, q)] = 1;
      }
      else{
        y[O_indexing(i, q)] = 0;
      }
    }
  }
  

	//Variable inialization
	Network_Setup();

    //Gather calibration samples from users

	for (i = 0; i < MOTIONS; i++){
		client.printf("6");
		client.printf("MOTION %d\r\n", i+1);
		while(1){
			Val();
			if (analyze == 1){
				client.printf("6");
				client.printf("Analyze Signal.\r\n");
				//Save all values to input
				client.printf("6");
				client.printf("Num : %d of %d\r\n", testNum+1, CAL_NUM);
				for (q = 0; q < FEAT; q++){
					if ((i == 0) && (testNum == 0)){
						min[q] = features[q];
					}
					else{
						if (features[q] < min[q]){
							min[q] = features[q];
						}
					}
					
					if (features[q] > max[q]){
						max[q] = features[q];
					}
					
				    input[I_indexing(i, testNum, q)] = features[q];
					features[q] = 0;
					
				}
				
				analyze = 0;
				count = 0; flag = 0; wait = 0;
				
				if (testNum >= (CAL_NUM-1)){
				    testNum = 0;
				    break;
				}
				testNum++;
			}
		}
	}

	for (i = 0; i < FEAT; i++){
		max_values[i] = max[i]*1.1;
		min_values[i] = min[i]*.9;
	}

	
	uint8_t count_loop = 0;
	client.printf("6");
    	client.printf("\r\n");
	client.printf("6");
	client.printf("Start Training...\r\n");
	while(1){
     for (i = 0; i < MOTIONS; i++){
		 //client.printf("Start of %d\r\n", i);
	      for (q = 0; q < n[i]; q++){
			  
             //client.printf("-----Loop index: %d %d-------\r\n", i, q);
			 
			  Input_init_cal(max_values, min_values, input, i, q);

	         feedforward(Layers);

	         backpropagation(Layers, y, LR, i);

			
			for (int p = 0; p < nodes[Layers-1]; p++){
				//client.printf("a (%d): %f\r\n", p, a[A_indexing(Layers-1, p)]);
				cost += -1 * y[O_indexing(i, p)] * log(a[A_indexing(Layers-1, p)]);
			}
			
	      } //----------------------END of cal num loop  
		  
       } //-------------------------END of MOTIONS loop
	   
		if ((abs(cost - last_cost) < .0005) && (cost < 10)) {
			break;
		}
		last_cost = cost;
		cost = 0;
	   count_loop++;
  } //--------------------------------END of while loop
  count_loop = 0;
  flag = 0; wait = 0; analyze = 0; count = 0;
  for (q = 0; q < FEAT; q++){
    features[q] = 0;
  }
  client.printf("6");
  client.printf("\r\n-------------------------------------------------------------------\r\n");
  client.printf("6");
  client.printf("Starting Operational Mode...\r\n\r\n");
}


/*
*** TITLE: Operational Function
*** FUNCTION: Using calibration output average features for each signal,
***				analyize and classify signal and send to robot subsystem. Learning
***				algorithm changes the signal feature averages based resulting signal.
***	INPUTS: NONE
*** OUTPUTS: NONE
*/
void Operational(){
  //variable
  uint8_t i = 0;
  float input[FEAT];
  uint8_t max_output = 0;
  uint8_t Layers = (sizeof(nodes) / sizeof(nodes[0]));
	//Get the adc value then check if valid
  while(1){
	   Val();

	    if (analyze == 1){
			for (i = 0; i < nodes[0]; i++){
				input[i] = features[i];
			}
			for (i = 0; i< FEAT; i++){
				features[i] = 0;
			}
			
			Input_init_Op(max_values, min_values, input);
			
		    feedforward(Layers);
			
		    for (uint8_t p = 0; p < nodes[Layers - 1]; p++){
				//client.printf("6");
		        //client.printf("Output (%d): %lf\r\n", p, a[A_indexing(Layers-1, p)]);
				if (a[A_indexing(Layers-1, p)] > a[A_indexing(Layers-1, max_output)]){
					max_output = p;
				}
		    }
			for (int m = 0; m < 1000; m++);
			if(a[A_indexing(Layers-1, max_output)] > .6){
				client.printf("%d", max_output+1);
			}
			else{
				client.printf("6");
				client.printf("Cannot be determined\r\n");
			}
			for (int m = 0; m < 1000; m++);
			//client.printf("6");
            //client.printf("--\r\n");
		    analyze = 0;
		    count = 0; flag = 0; wait = 0;
	      }
      }
}

/**********************************PRIVATE FUNCTIONS********************************************/

//Neural Network Code
float SigFunc(float i) {
	return (1 / (1 + (float)exp(-i)));
}

float softmax(float i) {
	return ((float)exp(i));
}

void Network_Setup() {
	//Setup Variables
	uint8_t M_size = 0;				//size of the matrix
	uint8_t i = 0;					//for loop index

	//Input node matrix
	M_size = sizeof(float)*A;
	memset(a, 0, M_size);
	//Output Node matrix
	M_size = sizeof(float)*B;
	memset(z, 0, M_size);
	//Error matrix
	memset(err, 0, M_size);
	//Bias Matrix
	memset(bias, 0, M_size);
	//weights matrix
	M_size = sizeof(float)*WEIGHTS_NUM;
	memset(weights, 0, M_size);

	//Inialize the weights and bias with random number
	//Future: possibily use clustering to determine initial values
	for (i = 0; i < WEIGHTS_NUM; i++) {
		weights[i] = /*(float)i/(float)100;*/(float)rand() / (float)RAND_MAX;
    //client.printf("weights inialized (%d): %f\r\n", i, weights[i]);
	}
}

void Network_Shutdown() {
	uint8_t M_size = 0;				//size of the matrix
	//Input node matrix
	M_size = sizeof(float)*A;
	memset(a, 0, M_size);
	//Output Node matrix
	M_size = sizeof(float)*B;
	memset(z, 0, M_size);
	//Error matrix
	memset(err, 0, M_size);
	//Bias Matrix
	memset(bias, 0, M_size);
	//weights matrix
	M_size = sizeof(float)*WEIGHTS_NUM;
	memset(weights, 0, M_size);
}

//input a - output -> z - output

void feedforward(uint8_t Layers) {
	//feedforward variables
	float temp = 0;											//temporary storage variable for calculations
	float softmax_temp = 0;
	uint8_t f = 0;													//for loop index
	uint8_t g = 0;													//..
	uint8_t n = 0;													//..

	for (f = 1; f < Layers; f++) {
		for (g = 0; g < nodes[f]; g++) {
			for (n = 0; n < nodes[f - 1]; n++) {
				temp += (weights[weight_indexing(f, n, g)] * a[A_indexing(f-1, n)]);
			}
			z[B_indexing(f, g)] = temp + bias[B_indexing(f, g)];
			if (f == (Layers-1)) {
				a[A_indexing(f, g)] = softmax(z[B_indexing(f, g)]);
				softmax_temp += a[A_indexing(f, g)];
			}
			else {
				a[A_indexing(f, g)] = SigFunc(z[B_indexing(f, g)]);
			}
			temp = 0;
		}
		if (f == (Layers - 1)) {
			for (g = 0; g < nodes[f]; g++) {
				a[A_indexing(f, g)] = a[A_indexing(f, g)] / softmax_temp;
			}
			softmax_temp = 0;
		}
	}
			
}



void backpropagation(uint8_t Layers, float y[], float LR, uint8_t ran) {

	float temp = 0;								//temporary calculation storage variable
	//float LR = (float).1;									//Learning Rate
	uint8_t train_size = 1;								//Training size
	uint8_t g = 0;										//for loop index
	uint8_t n = 0;										//..
	uint8_t f = 0;										//..

	for (g = 0; g < nodes[Layers - 1]; g++) {
		//client.printf("y[%d]: %f\r\n", g, y[O_indexing(ran, g)]);
		err[B_indexing(Layers-1, g)] = (a[A_indexing(Layers-1, g)] - y[O_indexing(ran, g)]);
    //client.printf("Error(%d, %d):%lf = (a(%d, %d):%lf - y(%d):%lf) * (o(%lf) * (1 - o(%lf)))\r\n", Layers - 1, g, err[B_indexing(Layers - 1, g)], Layers - 1, g, a[A_indexing(Layers - 1, g)], g, y[O_indexing(ran, g)], z[B_indexing(Layers - 1, g)], z[B_indexing(Layers - 1, g)]);
	}
	for (f = Layers - 2; f > 0; f--) {
		for (n = 0; n< nodes[f]; n++) {
			for (g = 0; g < nodes[f + 1]; g++) {
				temp += (weights[weight_indexing(f+1, n, g)] * err[B_indexing(f+1, g)]);
			}
			err[B_indexing(f, n)] = temp * (SigFunc(z[B_indexing(f, n)])*(1 - SigFunc(z[B_indexing(f, n)])));
			temp = 0;
		}
	}

	for (f = Layers - 1; f > 0; f--) {
		for (n = 0; n < nodes[f]; n++) {
			for (g = 0; g < nodes[f - 1]; g++) {
				weights[weight_indexing(f, g, n)] = weights[weight_indexing(f, g, n)] - ((LR / train_size)*(a[A_indexing(f-1, g)] * err[B_indexing(f, n)]));
			}
			bias[B_indexing(f, n)] = bias[B_indexing(f, n)] - ((LR / train_size) * err[B_indexing(f, n)]);
		}
	}
}


void Input_init_cal(float max_values[] , float min_values[], float input[], uint8_t motion, uint8_t cal) {
	for (uint8_t i = 0; i < nodes[0]; i++) {
		a[A_indexing(0, i)] = (input[I_indexing(motion, cal, i)] - min_values[i])/(max_values[i] - min_values[i]);
	}
}

void Input_init_Op(float max_values[] , float min_values[], float input[]) {
	for (uint8_t i = 0; i < nodes[0]; i++) {
		a[A_indexing(0, i)] = (input[i] - min_values[i])/(max_values[i] - min_values[i]);
	}
}


uint8_t A_indexing(uint8_t dimension, uint8_t node) {
	uint8_t p = 0;				//for loop index
	uint8_t index = 0;			//index for 1d array

	for (p = 0; p < dimension; p++) {
		index += nodes[p];
	}

	index += node;

	return index;
}

uint8_t B_indexing(uint8_t dimension, uint8_t node) {
	uint8_t index = 0;			//index for 1d array
	uint8_t p = 0;				//for loop index

	for (p = 1; p < dimension; p++) {
		index += nodes[p];
	}

	index += node;

	return index;
}

uint8_t weight_indexing(uint8_t dimension, uint8_t lnode, uint8_t node) {
	uint8_t index = 0;			//index for 1d array
	uint8_t p = 0;				//for loop index

	for (p = 1; p < dimension; p++) {
		index += (nodes[p]*nodes[p-1]);
	}

	index += lnode + (nodes[dimension-1] * node);

	return index;
}

//End Neural Network Code
//
//
//


void Val(){
    uint8_t i = 0;
    uint32_t val0 = 0;
    val0 = analogRead(analogPin0);

	if (!flag && val0 >= THRES && !wait){        		//Found Value to being segmenting
		SegAnalysis(val0, 0);
		flag = 1;
		count++;
	}
	//MOVE SIGNAL CONTINUED TOO LONG TO HERE?
	else if (flag && val0 < THRES && !wait && count >= MINLENGTH){             //End of segmentation - Now analyze the signal
		//client.printf("Signal End. count: %d\r\n", count);
		count++;
		SegAnalysis(val0, 1);
		flag = 0;
		analyze = 1;
	}
	else if (flag && val0 < THRES && !wait && count < MINLENGTH){
		//client.printf("Too Small. count: %d\r\n", count);
		for (i = 0; i < nodes[0]; i++){
			features[i] = 0;
		}
		count = 0;
		flag = 0;
		wait = 0;
		//client.printf("Too Short.\r\n");
	}
	else if (flag && count >= MAXLENGTH){      			//Signal has continued longer than max length cancel segment
		//client.printf("Too long. count: %d\r\n", count);
		for (i = 0; i < nodes[0]; i++){
			features[i] = 0;
		}
		count = 0;
		flag = 0;
		wait = 1;
		//client.printf("Too Long.\r\n");
	}
	else if (!flag && val0 < THRES && wait){    		//Waiting for signal to calm down after max length event
		//client.printf("Signal now calm.\r\n");
		wait = 0;
		count = 0;
	}
	else if (flag && val0 >= THRES && !wait){            //Add signal to segment
		//client.printf("Signal. count: %d\r\n", count);
		count++;
		SegAnalysis(val0, 0);
	}
}

void SegAnalysis(uint32_t val, uint8_t end){
  static uint32_t last_data = 0;

  //Features: RMS, Wave Complexity
  if (count == 0){

	//Calculate RMS
	features[0] += pow(val, 2);
	last_data = val;
  }
  else {


    //Calculate RMS
    features[0] += pow(val, 2);

    features[1] += fabs(val - last_data);
	
	last_data = val;

    if (end == 1){

      features[0] = sqrt(features[0]/count);
	  
	  //Reset static variables
	  last_data = 0;
    }

  }
}

inline uint8_t O_indexing(uint8_t i, uint8_t q){
  return ((MOTIONS*i) + q);
}

inline uint8_t I_indexing(uint8_t motion, uint8_t cal, uint8_t index){
  return ((motion * (CAL_NUM*FEAT)) + (cal * FEAT) + index);
}


