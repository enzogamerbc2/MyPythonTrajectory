import time

n1 = int(input("type a number: "))
n2 = int(input("type another number: "))
operation = input("type an operation (+, -, *, /): ")
result = 0

if operation == "+":
    result = n1 + n2
    print(f"{n1} + {n2} = {result}")
elif operation == "-":
    result = n1 - n2
    print(f"{n1} - {n2} = {result}")
elif operation == "*":
    result = n1 * n2
    print(f"{n1} * {n2} = {result}")
elif operation == "/":
    if n2 != 0:
        result = n1 / n2
        print(f"{n1} / {n2} = {result}")
    else:
        print("Error: Division by zero is not allowed.")
else:
    print("Invalid operation! Please choose from +, -, *, or / next time.")
    time.sleep(2)