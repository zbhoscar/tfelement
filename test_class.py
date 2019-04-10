class Dog(object):
    def eat(self, n):
        print('dog ate %d apples', n)
    def add_fn(self,a,b):
        return a+b

    def __init__(self, name, age):
        self.age = age
        self.name = name

    def show(self):
        print(self.name, self.age)

anb = Dog()
print(anb, type(anb))
anb.eat(4)
a = anb.add_fn(3, 7)
print(a)