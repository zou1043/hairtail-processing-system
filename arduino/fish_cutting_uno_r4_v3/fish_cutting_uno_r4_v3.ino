#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Servo.h>
#include <limits.h>
#include <math.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);
Servo flipServo;  // Servo 1: sorting gate on D6.

// =========================================================
// PCA9685 channel map
// =========================================================
// M1/M2 are still driven through TB6612 channels on PCA9685.
const int M1_PWM = 0;  const int M1_IN1 = 1;  const int M1_IN2 = 2;
const int M2_PWM = 3;  const int M2_IN1 = 4;  const int M2_IN2 = 5;

// High-power MOS boards. These are not TB6612 outputs.
const int SAW_PWM_CH  = 6;   // 775 saw MOS signal.
const int PUMP_PWM_CH = 7;   // Water pump MOS signal.

// =========================================================
// Arduino pins
// =========================================================
const int STBY_PIN = 2;
const int STEP_PIN = 3;      // Main screw stepper.
const int DIR_PIN  = 4;
const int EN_PIN   = 5;
const int SERVO_PIN = 6;     // Sorting servo.

const int LIMIT_HOME_PIN = 7; // Main screw home limit.
const int LIMIT_STOP_PIN = 8; // Fish stopper limit.

const int CUT_START_PIN = 9;  // New cutter screw start sensor, SN04-N on D9.
const int CUT_EN_PIN    = 10; // Cutter screw TB6600 ENA.
const int CUT_DIR_PIN   = 11; // Cutter screw TB6600 DIR.
const int CUT_STEP_PIN  = 12; // Cutter screw TB6600 PUL.
const int CUT_END_PIN   = A2; // New cutter screw end sensor, SN04-N on A2.

const int HX711_DT_PIN = A0;
const int HX711_SCK_PIN = A1;

// =========================================================
// Main screw / stepper params
// =========================================================
const float SCREW_PITCH_MM = 2.0;
const int MOTOR_STEPS_PER_REV = 200;
const int MICROSTEP = 16;
const float STEPS_PER_MM = (MOTOR_STEPS_PER_REV * MICROSTEP) / SCREW_PITCH_MM;

// ==== 新增：丝杠2 (横切丝杠) 专属参数 ====
const float CUT_SCREW_PITCH_MM = 5.0; // 1605丝杠的螺距是5mm
const int CUT_MICROSTEP = 8;          // TB6600 设置为 1600 Pulse/Rev (8细分)
const float CUT_STEPS_PER_MM = (MOTOR_STEPS_PER_REV * CUT_MICROSTEP) / CUT_SCREW_PITCH_MM;

const uint8_t HOME_DIR_LEVEL = HIGH;
const float MAX_TRAVEL_MM = 80.0;
float bladeOffsetMM = 45.0; // Distance from stopper to saw blade.

// Cutter screw direction levels. Reverse these two values if the real
// mechanism moves in the opposite direction.
const uint8_t CUT_FORWARD_DIR_LEVEL = LOW;
const uint8_t CUT_HOME_DIR_LEVEL    = HIGH;

// =========================================================
// Servo positions
// =========================================================
int SERVO_CENTER = 90;
int SERVO_HEAD = 0;
int SERVO_BODY = 180;

// =========================================================
// Runtime config
// =========================================================
int feedSpeed = 70;             // M2 conveyor speed.
int cutMotorSpeed = 85;         // Legacy name; now also used as saw MOS PWM percent.
int pumpCutSpeed = 0;           // Pump PWM percent during CUTTEST / automatic cut. Set via CFG PUMPSPEED.

unsigned long settleDelayMs = 200;
unsigned long sortHoldMs = 600;
unsigned long centerHoldMs = 250;
unsigned long ejectTailMs = 2000;
unsigned long cutMotorOnMs = 1200; // V6 meaning: dwell at A2 end sensor before returning.
unsigned long homeTimeoutMs = 30000;
unsigned long feedTimeoutMs = 30000;
unsigned long cutForwardTimeoutMs = 15000;
unsigned long cutHomeTimeoutMs = 15000;
unsigned int stepPulseUs = 80;
unsigned int cutStepPulseUs = 200;

float hx711Scale = 218.5f;
long hx711Offset = 0;
unsigned long hx711ReadyTimeoutMs = 500;

// =========================================================
// State variables
// =========================================================
volatile bool stopFlag = false;
float currentPosMM = 0.0;
String cmdBuffer = "";

struct CutTask {
  float lenMM;
  bool isHead;
};

const int MAX_TASKS = 24;
CutTask tasks[MAX_TASKS];
int taskCount = 0;
int currentTaskIndex = 0;

bool jobActive = false;

enum JobState {
  JOB_IDLE, JOB_INIT, JOB_FEED_TO_STOPPER, JOB_MOVE_CUT_LEN,
  JOB_CUT, JOB_SORT, JOB_SORT_HOLD, JOB_RETURN_CENTER,
  JOB_NEXT_TASK, JOB_EJECT_TAIL, JOB_FINISH, JOB_ERROR
};

JobState jobState = JOB_IDLE;
unsigned long stateTs = 0;
String lastError = "";

// =========================================================
// Forward declarations
// =========================================================
void serviceSerial();
void reportLimits();
void printJobStatus();
void handleCommand(String cmd);
void _handleCommandImpl(String cmd);
bool waitWithService(unsigned long ms);
bool homeCutScrewToStart(bool announceOk = true);
bool runCutCycle();

// =========================================================
// PCA / actuator helpers
// =========================================================
void setPCAChannelHigh(int ch) { pca.setPWM(ch, 0, 4095); }
void setPCAChannelLow(int ch)  { pca.setPWM(ch, 0, 0); }

void setPCAChannelPWMPercent(int ch, int percent) {
  percent = constrain(percent, 0, 100);
  uint16_t pwm = map(percent, 0, 100, 0, 4095);
  pca.setPWM(ch, 0, pwm);
}

void setMosOutput(int ch, int percent) {
  percent = constrain(percent, 0, 100);
  if (percent <= 0) {
    pca.setPWM(ch, 0, 0);
  } else {
    uint16_t pwmValue = map(percent, 0, 100, 0, 4095);
    pca.setPWM(ch, 0, pwmValue);
  }
}

void setSawOutput(int percent) {
  setMosOutput(SAW_PWM_CH, percent);
}

void setPumpOutput(int percent) {
  setMosOutput(PUMP_PWM_CH, percent);
}

void stopSawAndPump() {
  setSawOutput(0);
  setPumpOutput(0);
}

void setMotorChannels(int pwmCh, int in1Ch, int in2Ch, int speedPercent) {
  int speed = constrain(speedPercent, -100, 100);
  if (speed > 0) {
    setPCAChannelHigh(in1Ch);
    setPCAChannelLow(in2Ch);
    setPCAChannelPWMPercent(pwmCh, speed);
  } else if (speed < 0) {
    setPCAChannelLow(in1Ch);
    setPCAChannelHigh(in2Ch);
    setPCAChannelPWMPercent(pwmCh, -speed);
  } else {
    setPCAChannelLow(in1Ch);
    setPCAChannelLow(in2Ch);
    setPCAChannelLow(pwmCh);
  }
}

void setMotor(int motorId, int speedPercent) {
  digitalWrite(STBY_PIN, HIGH);
  switch (motorId) {
    case 1: setMotorChannels(M1_PWM, M1_IN1, M1_IN2, speedPercent); break;
    case 2: setMotorChannels(M2_PWM, M2_IN1, M2_IN2, speedPercent); break;
    default: break;
  }
}

void stopAllDCMotors() {
  setMotor(1, 0);
  setMotor(2, 0);
}

void enableStepper(bool enable) {
  digitalWrite(EN_PIN, enable ? LOW : HIGH);
}

void enableCutStepper(bool enable) {
  digitalWrite(CUT_EN_PIN, enable ? LOW : HIGH);
}

void pulseOnce(unsigned int usDelay) {
  digitalWrite(STEP_PIN, HIGH);
  delayMicroseconds(usDelay);
  digitalWrite(STEP_PIN, LOW);
  delayMicroseconds(usDelay);
}

void pulseCutOnce(unsigned int usDelay) {
  digitalWrite(CUT_STEP_PIN, HIGH);
  delayMicroseconds(usDelay);
  digitalWrite(CUT_STEP_PIN, LOW);
  delayMicroseconds(usDelay);
}

void stopAllActuators() {
  stopAllDCMotors();
  stopSawAndPump();
  enableStepper(false);
  enableCutStepper(false);
}

// =========================================================
// Servo 1 control
// =========================================================
void moveServoAngle(int angle) {
  angle = constrain(angle, 0, 180);
  flipServo.write(angle);
}

void moveServoNamed(const String &name) {
  if (name == "CENTER") {
    moveServoAngle(SERVO_CENTER);
    Serial.println("SERVO_OK CENTER");
  } else if (name == "HEAD" || name == "LEFT") {
    moveServoAngle(SERVO_HEAD);
    Serial.println("SERVO_OK HEAD");
  } else if (name == "BODY" || name == "RIGHT") {
    moveServoAngle(SERVO_BODY);
    Serial.println("SERVO_OK BODY");
  } else {
    Serial.println("ERR SERVO NAME");
  }
}

// =========================================================
// Sensor helpers
// =========================================================
bool limitTriggeredRaw(int pin) {
  return digitalRead(pin) == LOW; // INPUT_PULLUP, LOW = triggered.
}

bool limitTriggeredDebounced(int pin) {
  if (!limitTriggeredRaw(pin)) return false;
  delayMicroseconds(100);
  return limitTriggeredRaw(pin);
}

bool isHomeLimitTriggered() { return limitTriggeredDebounced(LIMIT_HOME_PIN); }
bool isStopLimitTriggered() { return limitTriggeredDebounced(LIMIT_STOP_PIN); }
bool isCutStartTriggered() { return limitTriggeredDebounced(CUT_START_PIN); }
bool isCutEndTriggered() { return limitTriggeredDebounced(CUT_END_PIN); }

void reportLimits() {
  Serial.print("LIMITS ");
  Serial.print("L1="); Serial.print(isHomeLimitTriggered() ? 1 : 0);
  Serial.print(" ");
  Serial.print("L2="); Serial.print(isStopLimitTriggered() ? 1 : 0);
  Serial.print(" ");
  Serial.print("CUT_START="); Serial.print(isCutStartTriggered() ? 1 : 0);
  Serial.print(" ");
  Serial.print("CUT_END="); Serial.println(isCutEndTriggered() ? 1 : 0);
}

// =========================================================
// HX711
// =========================================================
bool hx711Ready() {
  return digitalRead(HX711_DT_PIN) == LOW;
}

long readHX711RawOnce(unsigned long timeoutMs = 500) {
  unsigned long start = millis();
  while (!hx711Ready()) {
    serviceSerial();
    if (stopFlag) return LONG_MIN;
    if (millis() - start > timeoutMs) return LONG_MIN;
    delay(1);
  }

  unsigned long data = 0;
  noInterrupts();
  for (int i = 0; i < 24; i++) {
    digitalWrite(HX711_SCK_PIN, HIGH);
    delayMicroseconds(1);
    data = (data << 1) | (digitalRead(HX711_DT_PIN) ? 1UL : 0UL);
    digitalWrite(HX711_SCK_PIN, LOW);
    delayMicroseconds(1);
  }
  digitalWrite(HX711_SCK_PIN, HIGH);
  delayMicroseconds(1);
  digitalWrite(HX711_SCK_PIN, LOW);
  interrupts();

  if (data & 0x800000UL) {
    data |= 0xFF000000UL;
  }
  return (long)data;
}

void reportWeight() {
  const int samples = 5;
  long sum = 0;
  int okCount = 0;

  for (int i = 0; i < samples; i++) {
    long raw = readHX711RawOnce(hx711ReadyTimeoutMs);
    if (raw != LONG_MIN) {
      sum += raw;
      okCount++;
    }
    delay(2);
  }

  if (okCount <= 0) {
    Serial.println("WEIGHT_ERR TIMEOUT");
    return;
  }

  float avgRaw = (float)sum / okCount;
  float weightG = (avgRaw - hx711Offset) / hx711Scale;
  Serial.print("WEIGHT=");
  Serial.println(weightG, 1);
}

void tareHX711() {
  const int samples = 10;
  long sum = 0;
  int okCount = 0;

  for (int i = 0; i < samples; i++) {
    long raw = readHX711RawOnce(hx711ReadyTimeoutMs);
    if (raw != LONG_MIN) {
      sum += raw;
      okCount++;
    }
    delay(2);
  }

  if (okCount <= 0) {
    Serial.println("WEIGHT_ERR TARE_TIMEOUT");
    return;
  }

  hx711Offset = sum / okCount;
  Serial.print("TARE_OK OFFSET=");
  Serial.println(hx711Offset);
}

// =========================================================
// Wait / motion helpers
// =========================================================
bool waitWithService(unsigned long ms) {
  unsigned long start = millis();
  while (millis() - start < ms) {
    serviceSerial();
    if (stopFlag) return false;
    delay(1);
  }
  return true;
}

bool moveStepperRelativeMM(float distMM, unsigned int pulseUs) {
  if (fabs(distMM) < 0.001f) {
    Serial.print("DONE STEP POS=");
    Serial.println(currentPosMM, 3);
    return true;
  }

  float targetPos = currentPosMM + distMM;
  if (targetPos < -0.01f || targetPos > MAX_TRAVEL_MM) {
    Serial.print("ERR SOFT_LIMIT_EXCEEDED TARGET=");
    Serial.println(targetPos, 2);
    return false;
  }

  long totalSteps = lround(fabs(distMM) * STEPS_PER_MM);
  bool forward = distMM >= 0;
  digitalWrite(DIR_PIN, forward ? (HOME_DIR_LEVEL == LOW ? HIGH : LOW) : HOME_DIR_LEVEL);
  enableStepper(true);

  for (long i = 0; i < totalSteps; i++) {
    serviceSerial();
    if (stopFlag) {
      enableStepper(false);
      Serial.println("STOPPED");
      return false;
    }
    if (!forward && limitTriggeredRaw(LIMIT_HOME_PIN)) {
      currentPosMM = 0.0;
      enableStepper(false);
      Serial.println("ERR LIMIT_HOME_HIT");
      return false;
    }
    pulseOnce(pulseUs);
  }

  currentPosMM = targetPos;
  if (currentPosMM < 0.001f) currentPosMM = 0.0;
  enableStepper(false);
  Serial.print("DONE STEP POS=");
  Serial.println(currentPosMM, 3);
  return true;
}

bool homeToPhysicalZero(unsigned int pulseUs) {
  digitalWrite(DIR_PIN, HOME_DIR_LEVEL);
  enableStepper(true);
  unsigned long start = millis();

  while (!isHomeLimitTriggered()) {
    serviceSerial();
    if (stopFlag) {
      enableStepper(false);
      Serial.println("STOPPED");
      return false;
    }
    if (millis() - start > homeTimeoutMs) {
      enableStepper(false);
      Serial.println("ERR HOME_TIMEOUT");
      return false;
    }
    pulseOnce(pulseUs);
  }

  enableStepper(false);
  currentPosMM = 0.0;
  Serial.println("HOME_OK POS=0.000");
  return true;
}

bool feedUntilStopLimit() {
  setMotor(2, feedSpeed);
  setMotor(1, cutMotorSpeed);
  unsigned long start = millis();

  while (!isStopLimitTriggered()) {
    serviceSerial();
    if (stopFlag) {
      stopAllDCMotors();
      Serial.println("STOPPED");
      return false;
    }
    if (millis() - start > feedTimeoutMs) {
      stopAllDCMotors();
      Serial.println("JOBERROR FEED_TIMEOUT");
      return false;
    }
    delay(1);
  }

  stopAllDCMotors();
  if (!waitWithService(settleDelayMs)) {
    return false;
  }
  Serial.println("FEEDSTOP_OK");
  return true;
}

bool moveCutStepperRelativeMM(float distMM, unsigned int pulseUs) {
  if (fabs(distMM) < 0.001f) return true;
  long totalSteps = lround(fabs(distMM) * CUT_STEPS_PER_MM);
  bool forward = distMM >= 0;
  digitalWrite(CUT_DIR_PIN, forward ? CUT_FORWARD_DIR_LEVEL : CUT_HOME_DIR_LEVEL);
  enableCutStepper(true);
  for (long i = 0; i < totalSteps; i++) {
    serviceSerial();
    if (stopFlag) {
      enableCutStepper(false);
      Serial.println("STOPPED");
      return false;
    }
    pulseCutOnce(pulseUs);
  }
  enableCutStepper(false);
  return true;
}

bool moveCutScrewUntilSensor(int sensorPin, uint8_t dirLevel, unsigned long timeoutMs, const char *timeoutCode) {
  digitalWrite(CUT_DIR_PIN, dirLevel);
  enableCutStepper(true);
  unsigned long start = millis();

  while (!limitTriggeredDebounced(sensorPin)) {
    serviceSerial();
    if (stopFlag) {
      enableCutStepper(false);
      return false;
    }
    if (millis() - start > timeoutMs) {
      enableCutStepper(false);
      Serial.print("CUT_ERR ");
      Serial.println(timeoutCode);
      return false;
    }
    pulseCutOnce(cutStepPulseUs);
  }

  digitalWrite(CUT_STEP_PIN, LOW);
  return true;
}

bool homeCutScrewToStart(bool announceOk) {
  if (isCutStartTriggered()) {
    enableCutStepper(false);
    if (announceOk) {
      Serial.println("CUTHOME_OK");
      reportLimits();
    }
    return true;
  }

  bool ok = moveCutScrewUntilSensor(CUT_START_PIN, CUT_HOME_DIR_LEVEL, cutHomeTimeoutMs, "HOME_TIMEOUT");
  enableCutStepper(false);
  if (!ok) {
    if (stopFlag) Serial.println("STOPPED");
    return false;
  }

  if (announceOk) {
    Serial.println("CUTHOME_OK");
    reportLimits();
  }
  return true;
}

bool runCutCycle() {
  if (!isCutStartTriggered()) {
    Serial.println("CUT_ERR START_NOT_HOME");
    return false;
  }
  if (isCutEndTriggered()) {
    Serial.println("CUT_ERR END_ACTIVE_AT_START");
    return false;
  }

  setSawOutput(cutMotorSpeed);
  setPumpOutput(pumpCutSpeed);

  bool ok = moveCutScrewUntilSensor(CUT_END_PIN, CUT_FORWARD_DIR_LEVEL, cutForwardTimeoutMs, "END_TIMEOUT");
  if (ok) {
    ok = waitWithService(cutMotorOnMs);
  }
  if (ok) {
    ok = homeCutScrewToStart(false);
  }

  enableCutStepper(false);
  stopSawAndPump();

  if (!ok) {
    if (stopFlag) Serial.println("STOPPED");
    return false;
  }

  Serial.println("CUT_OK");
  reportLimits();
  return true;
}

bool ejectTailForMs(unsigned long runMs) {
  setMotor(2, feedSpeed);
  setMotor(1, cutMotorSpeed);
  bool ok = waitWithService(runMs);
  stopAllDCMotors();
  if (ok) {
    Serial.println("EJECT_OK");
    return true;
  }
  Serial.println("STOPPED");
  return false;
}

// =========================================================
// Job state machine
// =========================================================
void jobAbort(const String &reason) {
  stopAllActuators();
  jobActive = false;
  jobState = JOB_ERROR;
  lastError = reason;
  Serial.print("JOBERROR ");
  Serial.println(reason);
}

void setJobState(JobState newState, const String &label) {
  jobState = newState;
  stateTs = millis();
  Serial.print("JOBSTATE ");
  Serial.println(label);
}

void clearTasks() {
  taskCount = 0;
  currentTaskIndex = 0;
  jobActive = false;
  jobState = JOB_IDLE;
  lastError = "";
  Serial.println("JOBCLEARED");
}

void printJobStatus() {
  Serial.print("JOBSTATUS active="); Serial.print(jobActive ? 1 : 0);
  Serial.print(" state="); Serial.print((int)jobState);
  Serial.print(" idx="); Serial.print(currentTaskIndex + 1);
  Serial.print("/"); Serial.print(taskCount);
  Serial.print(" pos="); Serial.print(currentPosMM, 3);
  Serial.print(" cutStart="); Serial.print(isCutStartTriggered() ? 1 : 0);
  Serial.print(" cutEnd="); Serial.println(isCutEndTriggered() ? 1 : 0);
}

bool commandAllowedWhileJobActive(const String &cmd) {
  return cmd == "STOP" || cmd == "LIMITS" || cmd == "JOBSTATUS";
}

void handleCommand(String cmd) {
  static bool inHandleCommand = false;
  cmd.trim();
  if (cmd.length() == 0) return;

  if (inHandleCommand) {
    if (cmd == "STOP") {
      stopFlag = true;
      stopAllActuators();
      jobActive = false;
      Serial.println("STOPPED");
    } else if (cmd == "LIMITS") {
      reportLimits();
    } else if (cmd == "JOBSTATUS") {
      printJobStatus();
    } else {
      Serial.print("ERR RECURSION_BLOCKED ");
      Serial.println(cmd);
    }
    return;
  }

  inHandleCommand = true;
  _handleCommandImpl(cmd);
  inHandleCommand = false;
}

void _handleCommandImpl(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd == "HOME" || cmd == "ZERO" || cmd == "GOTOZERO" || cmd == "FEEDSTOP" ||
      cmd == "JOBSTART" || cmd == "CUTTEST" || cmd == "CUTHOME" ||
      cmd == "CUTFORWARD" || cmd == "CUTREVERSE" ||
      cmd.startsWith("STEP ") || cmd.startsWith("CUTSTEP ") || cmd.startsWith("EJECT ") ||
      cmd.startsWith("ALL ") || cmd.startsWith("M1 ") || cmd.startsWith("M2 ") ||
      cmd.startsWith("FLIP") || cmd.startsWith("SERVO ")) {
    stopFlag = false;
  }

  if (cmd == "STOP") {
    stopFlag = true;
    stopAllActuators();
    jobActive = false;
    Serial.println("STOPPED");
    return;
  }

  if (jobActive && !commandAllowedWhileJobActive(cmd)) {
    Serial.print("ERR BUSY ");
    Serial.println(cmd);
    return;
  }

  if (cmd == "LIMITS") { reportLimits(); return; }
  if (cmd == "HOME" || cmd == "ZERO") { homeToPhysicalZero(stepPulseUs); return; }
  if (cmd == "SETZERO") { currentPosMM = 0.0; Serial.println("ZERO SET"); return; }
  if (cmd == "GOTOZERO") {
    float distToZero = -currentPosMM;
    moveStepperRelativeMM(distToZero, stepPulseUs);
    return;
  }
  if (cmd == "JOBSTATUS") { printJobStatus(); return; }

  if (cmd == "CUTHOME") {
    homeCutScrewToStart(true);
    return;
  }

  if (cmd == "CUTTEST") {
    runCutCycle();
    return;
  }

  if (cmd == "CUTFORWARD") {
    bool ok = moveCutStepperRelativeMM(10.0, cutStepPulseUs);
    if (ok) {
      Serial.println("CUT_FORWARD_OK");
      reportLimits();
    } else if (stopFlag) {
      Serial.println("STOPPED");
    }
    return;
  }

  if (cmd == "CUTREVERSE") {
    bool ok = moveCutStepperRelativeMM(-10.0, cutStepPulseUs);
    if (ok) {
      Serial.println("CUT_REVERSE_OK");
      reportLimits();
    } else if (stopFlag) {
      Serial.println("STOPPED");
    }
    return;
  }

  if (cmd.startsWith("STEP ")) {
    moveStepperRelativeMM(cmd.substring(5).toFloat(), stepPulseUs);
    return;
  }

  if (cmd.startsWith("CUTSTEP ")) {
    float dist = cmd.substring(8).toFloat();
    bool ok = moveCutStepperRelativeMM(dist, cutStepPulseUs);
    if (ok) {
      Serial.println("CUT_STEP_OK");
      reportLimits();
    } else if (stopFlag) {
      Serial.println("STOPPED");
    }
    return;
  }

  if (cmd.startsWith("ALL ")) {
    int speed = cmd.substring(4).toInt();
    setMotor(1, speed);
    setMotor(2, speed);
    Serial.println("OK ALL");
    return;
  }
  if (cmd.startsWith("M1 ")) { setMotor(1, cmd.substring(3).toInt()); Serial.println("OK M1"); return; }
  if (cmd.startsWith("M2 ")) { setMotor(2, cmd.substring(3).toInt()); Serial.println("OK M2"); return; }

  if (cmd.startsWith("SAW ")) {
    int speed = constrain(cmd.substring(4).toInt(), 0, 100);
    setSawOutput(speed);
    Serial.println("OK SAW");
    return;
  }

  if (cmd.startsWith("PUMP ")) {
    int speed = constrain(cmd.substring(5).toInt(), 0, 100);
    setPumpOutput(speed);
    Serial.println("OK PUMP");
    return;
  }

  if (cmd.startsWith("FLIP ")) {
    int angle = cmd.substring(5).toInt();
    moveServoAngle(angle);
    Serial.println("OK FLIP");
    return;
  }

  // V6 compatibility only: the old servo2 mechanism is removed because D9 is now CUT_START.
  if (cmd.startsWith("FLIP2 ")) {
    Serial.println("OK FLIP2 IGNORED");
    return;
  }

  if (cmd.startsWith("SERVO ")) {
    String name = cmd.substring(6);
    name.trim();
    name.toUpperCase();
    moveServoNamed(name);
    return;
  }

  if (cmd == "FEEDSTOP") {
    if (feedUntilStopLimit()) Serial.println("OK FEEDSTOP");
    return;
  }

  if (cmd == "WEIGHT") {
    reportWeight();
    return;
  }

  if (cmd == "TARE") {
    tareHX711();
    return;
  }

  if (cmd.startsWith("EJECT ")) {
    unsigned long ms = cmd.substring(6).toInt();
    ejectTailForMs(ms);
    return;
  }

  if (cmd == "JOBCLEAR") {
    clearTasks();
    return;
  }

  if (cmd.startsWith("JOBADD ")) {
    if (taskCount >= MAX_TASKS) {
      Serial.println("ERR JOB FULL");
      return;
    }
    int sp1 = cmd.indexOf(' ');
    int sp2 = cmd.indexOf(' ', sp1 + 1);
    if (sp2 < 0) {
      Serial.println("ERR JOBADD FORMAT");
      return;
    }
    float len = cmd.substring(sp1 + 1, sp2).toFloat();
    String kind = cmd.substring(sp2 + 1);
    kind.trim();
    kind.toUpperCase();
    tasks[taskCount].lenMM = len;
    tasks[taskCount].isHead = (kind == "HEAD");
    taskCount++;
    Serial.print("JOBADD_OK count=");
    Serial.println(taskCount);
    return;
  }

  if (cmd == "JOBSTART") {
    if (taskCount <= 0) {
      Serial.println("JOBERROR NO_TASKS");
      return;
    }
    currentTaskIndex = 0;
    jobActive = true;
    setJobState(JOB_INIT, "INIT");
    return;
  }

  if (cmd.startsWith("CFG ")) {
    int s1 = cmd.indexOf(' ');
    int s2 = cmd.indexOf(' ', s1 + 1);
    if (s2 < 0) {
      Serial.println("ERR CFG FORMAT");
      return;
    }
    String key = cmd.substring(s1 + 1, s2);
    String value = cmd.substring(s2 + 1);
    key.toUpperCase();
    int v = value.toInt();
    float fv = value.toFloat();

    if (key == "FEED") feedSpeed = constrain(v, -100, 100);
    else if (key == "CUTSPEED" || key == "SAWSPEED") cutMotorSpeed = constrain(v, 0, 100);
    else if (key == "PUMP" || key == "PUMPSPEED") pumpCutSpeed = constrain(v, 0, 100);
    else if (key == "CUTTIME" || key == "CUTDWELL") cutMotorOnMs = (unsigned long) max(v, 0);
    else if (key == "CUTSTEPUS") cutStepPulseUs = (unsigned int) max(v, 1);
    else if (key == "CUTTIMEOUT") cutForwardTimeoutMs = (unsigned long) max(v, 100);
    else if (key == "CUTHOMETIMEOUT") cutHomeTimeoutMs = (unsigned long) max(v, 100);
    else if (key == "OFFSET" || key == "BLADEOFFSET") bladeOffsetMM = fv;
    else if (key == "SETTLE") settleDelayMs = (unsigned long) max(v, 0);
    else if (key == "SORTHOLD") sortHoldMs = (unsigned long) max(v, 0);
    else if (key == "CENTERHOLD") centerHoldMs = (unsigned long) max(v, 0);
    else if (key == "EJECT") ejectTailMs = (unsigned long) max(v, 0);
    else if (key == "SERVOCENTER") SERVO_CENTER = constrain(v, 0, 180);
    else if (key == "SERVOHEAD") SERVO_HEAD = constrain(v, 0, 180);
    else if (key == "SERVOBODY") SERVO_BODY = constrain(v, 0, 180);
    else if (key == "SERVO2DOWN" || key == "SERVO2UP") {
      // Ignored for V6, accepted so older backends can still dispatch tasks during migration.
    }
    else {
      Serial.println("ERR CFG KEY");
      return;
    }
    Serial.println("CFG_OK");
    return;
  }

  Serial.println("ERR UNKNOWN CMD");
}

void serviceSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(cmdBuffer);
      cmdBuffer = "";
    } else if (c != '\r') {
      cmdBuffer += c;
      if (cmdBuffer.length() > 96) {
        cmdBuffer = "";
        Serial.println("ERR CMD TOO_LONG");
      }
    }
  }
}

void processJobState() {
  if (!jobActive) return;

  switch (jobState) {
    case JOB_INIT:
      moveServoNamed("CENTER");
      if (!waitWithService(centerHoldMs)) { jobAbort("STOP"); return; }
      if (!homeToPhysicalZero(stepPulseUs)) { jobAbort("HOME_FAIL"); return; }
      if (!homeCutScrewToStart(false)) { jobAbort("CUT_HOME_FAIL"); return; }
      setJobState(JOB_FEED_TO_STOPPER, "FEED_TO_STOPPER");
      break;

    case JOB_FEED_TO_STOPPER:
      if (currentTaskIndex >= taskCount) {
        setJobState(JOB_EJECT_TAIL, "EJECT_TAIL");
        break;
      }
      Serial.print("JOBSTATE FEED idx=");
      Serial.print(currentTaskIndex + 1);
      Serial.print("/");
      Serial.print(taskCount);
      Serial.print(" len=");
      Serial.print(tasks[currentTaskIndex].lenMM, 2);
      Serial.print(" kind=");
      Serial.println(tasks[currentTaskIndex].isHead ? "HEAD" : "BODY");
      if (!feedUntilStopLimit()) { jobAbort("FEEDSTOP_FAIL"); return; }
      jobState = JOB_MOVE_CUT_LEN;
      break;

    case JOB_MOVE_CUT_LEN: {
      Serial.print("JOBSTATE MOVE_CUT idx=");
      Serial.print(currentTaskIndex + 1);
      
      // 1. 恢复：计算实际推料距离（减去挡板到刀片的物理偏移量）
      float actualMoveDist = tasks[currentTaskIndex].lenMM - bladeOffsetMM;
      if (actualMoveDist < 0) actualMoveDist = 0; // 防呆保护
      
      Serial.print(" target="); Serial.print(tasks[currentTaskIndex].lenMM, 2);
      Serial.print(" move="); Serial.println(actualMoveDist, 2);

      // 2. 恢复：开启传送带和刷鳞电机，低速温柔同步推鱼
      int syncSpeed = 40; 
      setMotor(2, syncSpeed);       
      setMotor(1, syncSpeed);

      // 执行丝杠推进
      if (!moveStepperRelativeMM(actualMoveDist, stepPulseUs)) { 
        stopAllDCMotors(); // 如果急停或撞限位，立刻切断电机
        jobAbort("STEP_FAIL"); 
        return; 
      }
      
      stopAllDCMotors(); // 到位后立刻刹车停下传送带
      jobState = JOB_CUT;
      break;
    }

    case JOB_CUT:
      Serial.print("JOBSTATE CUT idx=");
      Serial.println(currentTaskIndex + 1);
      if (!runCutCycle()) { jobAbort("CUT_FAIL"); return; }
      jobState = JOB_SORT;
      break;

    case JOB_SORT:
      if (tasks[currentTaskIndex].isHead) {
        Serial.print("JOBSTATE SORT HEAD idx=");
        Serial.println(currentTaskIndex + 1);
        moveServoNamed("HEAD");
      } else {
        Serial.print("JOBSTATE SORT BODY idx=");
        Serial.println(currentTaskIndex + 1);
        moveServoNamed("BODY");
      }
      stateTs = millis();
      jobState = JOB_SORT_HOLD;
      break;

    case JOB_SORT_HOLD:
      if (millis() - stateTs >= sortHoldMs) {
        Serial.println("JOBSTATE RETURN_CENTER");
        moveServoNamed("CENTER");
        stateTs = millis();
        jobState = JOB_RETURN_CENTER;
      }
      break;

    case JOB_RETURN_CENTER:
      if (millis() - stateTs >= centerHoldMs) {
        Serial.println("JOBSTATE RETURN_HOME");
        if (!homeToPhysicalZero(stepPulseUs)) { jobAbort("RETURN_HOME_FAIL"); return; }
        jobState = JOB_NEXT_TASK;
      }
      break;

    case JOB_NEXT_TASK:
      currentTaskIndex++;
      if (currentTaskIndex < taskCount) {
        setJobState(JOB_FEED_TO_STOPPER, "NEXT_CUT");
      } else {
        setJobState(JOB_EJECT_TAIL, "EJECT_TAIL");
      }
      break;

    case JOB_EJECT_TAIL:
      if (!ejectTailForMs(ejectTailMs)) { jobAbort("EJECT_FAIL"); return; }
      setJobState(JOB_FINISH, "FINISH");
      break;

    case JOB_FINISH:
      moveServoNamed("CENTER");
      if (!homeToPhysicalZero(stepPulseUs)) { jobAbort("FINAL_HOME_FAIL"); return; }
      if (!homeCutScrewToStart(false)) { jobAbort("FINAL_CUT_HOME_FAIL"); return; }
      jobActive = false;
      jobState = JOB_IDLE;
      Serial.println("JOBDONE");
      break;

    case JOB_ERROR:
      jobActive = false;
      break;

    case JOB_IDLE:
    default:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pca.begin();
  pca.setPWMFreq(1000);

  pinMode(STBY_PIN, OUTPUT);
  digitalWrite(STBY_PIN, HIGH);

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);
  enableStepper(false);

  pinMode(CUT_EN_PIN, OUTPUT);
  pinMode(CUT_DIR_PIN, OUTPUT);
  pinMode(CUT_STEP_PIN, OUTPUT);
  digitalWrite(CUT_STEP_PIN, LOW);
  enableCutStepper(false);

  pinMode(LIMIT_HOME_PIN, INPUT_PULLUP);
  pinMode(LIMIT_STOP_PIN, INPUT_PULLUP);
  pinMode(CUT_START_PIN, INPUT_PULLUP);
  pinMode(CUT_END_PIN, INPUT_PULLUP);

  pinMode(HX711_DT_PIN, INPUT);
  pinMode(HX711_SCK_PIN, OUTPUT);
  digitalWrite(HX711_SCK_PIN, LOW);

  flipServo.attach(SERVO_PIN);
  moveServoAngle(SERVO_CENTER);

  stopAllActuators();
  Serial.println("UNO R4 READY V6 CLOSED_LOOP_CUT");
  reportLimits();
}

void loop() {
  serviceSerial();
  processJobState();
}
