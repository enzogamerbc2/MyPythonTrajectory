import random
import time
import os

number = random.randint(1, 10)
attempts = 0
guess = 0

def guesses():

    global guess
    guess = int(input("Guess the number between 1 - 10: "))

while guess != number:

    attempts += 1
    print("Wrong number!")
    print(f"Attempts: {attempts}")
    guesses()

else:

    print("You guessed it!")
    time.sleep(3)
