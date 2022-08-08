import turtle
import turtle as t
import time

def true():
    screen = turtle.Screen()
    screen.setup(375, 700)

    circle = turtle.Turtle()
    circle.shape('circle')
    circle.color('red')
    circle.speed('fastest')
    circle.up()

    square = turtle.Turtle()
    square.shape('square')
    square.color('green')
    square.speed('fastest')
    square.up()

    circle.goto(0, 280)
    circle.stamp()

    k = 0
    for i in range(1, 13):
        y = 30 * i
        for j in range(i - k):
            x = 30 * j
            square.goto(x, -y + 280)
            square.stamp()
            square.goto(-x, -y + 280)
            square.stamp()

        if i % 4 == 0:
            x = 30 * (j + 1)
            circle.color('red')
            circle.goto(-x, -y + 280)
            circle.stamp()
            circle.goto(x, -y + 280)
            circle.stamp()
            k += 3

        if i % 4 == 3:
            x = 30 * (j + 1)
            circle.color('yellow')
            circle.goto(-x, -y + 280)
            circle.stamp()
            circle.goto(x, -y + 280)
            circle.stamp()

    square.color('brown')
    for i in range(13, 200):
        y = 30 * i
        for j in range(2):
            x = 30 * j
            square.goto(x, -y + 280)
            square.stamp()
            square.goto(-x, -y + 280)
            square.stamp()

    text = turtle.Turtle()
    text.hideturtle()
    text.penup()
    text.goto(-120, 270)
    text.color('red')
    text.write('Wish: Merry Christmas', font=('Arial', 15, 'bold'), align="center")
    #text.write('smooth postgraduate entrance examination', font=('Arial', 15, 'bold'), align="center")
def initdata():
    t.setup(800,600)
    t.pencolor('red')
    t.pensize(5)
    t.speed(10)

def move_pen(x,y):
    t.hideturtle()
    t.up()
    t.goto(x,y)
    t.down()
    t.showturtle()

def hart_arc():
    for i in range(200):
        t.right(1)
        t.forward(2)

def draw():
    name=input("请输入姓名：")
    sign=input("请输入你的大名：")
    initdata()
    move_pen(0,-180)
    t.left(140)
    t.fillcolor("pink")
    t.begin_fill()
    t.forward(224)
    hart_arc()
    t.left(120)
    hart_arc()
    t.forward(224)
    t.end_fill()
    move_pen(x=70, y=160)
    t.left(185)
    t.circle(-110,185)
    t.forward(50)
    move_pen(-180,-180)
    t.left(180)
    t.forward(600)
    move_pen(0,50)
    t.hideturtle()
    t.color('#CD5C5C', 'red')
    t.write(name, font=('Arial', 20, 'bold'), align="center")
    t.color('red', 'pink')
    time.sleep(2)
    move_pen(220, -180)
    t.hideturtle()
    t.write(sign, font=('Arial', 18, 'bold'), align="center")
def main():
    draw()
    time.sleep(5)
if __name__ == '__main__':
    true()
    main()
