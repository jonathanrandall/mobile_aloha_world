#ifndef MOTOR_DRIVE_STUFF_H_
#define MOTOR_DRIVE_STUFF_H_

#include "wheel_encoder_stuff.h"

int gamma1 = 0;
// Motor right

int enA = 13; //white
int enB = 33; //yellow
int max_spd = 255;
int min_spd = 60;

int rmI1 = 12; //green
int rmI2 = 27; //blue
int lmI1 = 25; // H out1 = H | L out1 = L | H short break | L stop purple
int lmI2 = 26; // L out2 = L | H out2 = H | H short break | L stop brown
// int STDBY = 19; //Low is off (current save) //not connected so far


bool robot_moving=false;

int noStop = 0;

//Setting Motor PWM properties
long int freq = 2000;
const int motorPWMChannnelA = 7;
const int motorPWMChannnelB = 2;
uint8_t lresolution = 8;
 
uint8_t  motor_speed   = 100;
volatile unsigned long previous_time = 0;
volatile unsigned long move_interval = 250;



void update_speed()
{ 
  ledcWrite(motorPWMChannnelA, motor_speed);

//    Serial.println("motor_speed");
    
  ledcWrite(motorPWMChannnelB, motor_speed+gamma1);

//    Serial.println(motor_speed);
    
}


enum state {fwd, back, stp, right, left};
state actstate = stp;
uint8_t robo = 0;

void robot_stop()
{
  digitalWrite(lmI1,HIGH);
  digitalWrite(lmI2,HIGH);
  digitalWrite(rmI1,HIGH);
  digitalWrite(rmI2,HIGH);
  actstate = stp;
}

void robot_fwd()
{
  if(actstate != fwd){
    digitalWrite(lmI1,HIGH);
    digitalWrite(lmI2,LOW);
    digitalWrite(rmI1,HIGH);
    digitalWrite(rmI2,LOW);
  move_interval=250;
  actstate = fwd;
  }
  previous_time = millis();  
  
}
 
void robot_back()
{
  if(actstate != back){
    digitalWrite(lmI1,LOW);
    digitalWrite(lmI2,HIGH);
    digitalWrite(rmI1,LOW);
    digitalWrite(rmI2,HIGH);
  move_interval=250;
  actstate = back;
  }
   previous_time = millis();  
}
 
void robot_left()
{
  if(actstate != left){
    digitalWrite(lmI1,LOW);
    digitalWrite(lmI2,HIGH);
    digitalWrite(rmI1,HIGH);
    digitalWrite(rmI2,LOW);
  move_interval=100;
  actstate = left;
  }
   previous_time = millis();
}
 
void robot_right()
{
  if(actstate != right){
    digitalWrite(lmI1,HIGH);
    digitalWrite(lmI2,LOW);
    digitalWrite(rmI1,LOW);
    digitalWrite(rmI2,HIGH);
  move_interval=100;
  actstate = right;
  }
   previous_time = millis();
}

void robot_setup(){

  // Pins for Motor Controller
    pinMode(lmI1,OUTPUT);
    pinMode(lmI2,OUTPUT);
    pinMode(rmI1,OUTPUT);
    pinMode(rmI2,OUTPUT);
    
    // Make sure we are stopped
    robot_stop();
 
    // Motor uses PWM Channel 8
    
    delay(100);
    
    ledcSetup(motorPWMChannnelA, freq, lresolution);
    ledcSetup(motorPWMChannnelB, freq, lresolution);
    
    
    
    ledcAttachPin(enA, motorPWMChannnelA);
    ledcAttachPin(enB, motorPWMChannnelB);

    // ledcWrite(motorPWMChannnelA, 130);
//    ledcWrite(motorPWMChannnelB, 130);
    
  
}

void robot_move(long *enc){
  robot_moving = true;
  Serial.println("in robot moving");
  unsigned int l1, r1;
  l1 = (unsigned int) enc[0];
  r1 = (unsigned int) enc[1];
  update_speed();

  bool f1 = (bool) enc[2];
  bool b1 = (bool) enc[3];

  Serial.println(String(f1)+" "+String(b1));

  while ((counter1+counter2) < (l1 + r1)){
    Serial.println(counter1);
    Serial.println(counter2);
    Serial.println("l1");
    Serial.println(String(l1) + " " + String(r1));
    delay(100);
    // counter1++;
    if (!f1){
      Serial.println("moving fwsd");
      robot_fwd();
    } 
    else { 
      // if (!b1) 
      // {
        robot_back();
      // }
    }
  }
  delay(1);
  Serial.println("still in robot moving0");
  robot_stop();
  delay(1);
  Serial.println("still in robot moving");
  reset_wheel_encoder_data();

  
  robot_moving=false;
  
}

void robot_move_loop(void *params){
  int spd_prev=0;
  long rec_cmd[4];

  while(true){
    if (spd_prev!=motor_speed){
      update_speed();
      spd_prev = motor_speed;
    }

    if(xQueueReceive(cmd_queue, &rec_cmd, pdMS_TO_TICKS(1))== pdTRUE) {
      
      while ((counter1+counter2) < (rec_cmd[0] + rec_cmd[1])){     
        if(!rec_cmd[2]){
              robot_fwd();
        } else if(!rec_cmd[3]){  //move in reverse
          robot_back();
        } else {
          break;
        }
      }
      // robot_stop();
    } else{
      robot_stop();
      reset_wheel_encoder_data();
    }
    

  }

}





#endif