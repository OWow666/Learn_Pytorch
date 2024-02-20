class Person:
    def __call__(self, name):
        print("__call__"+"Hello"+name)

    def hell(self, name):
        print("Hello")

person = Person()
person("zhangsan")
person.hell("lisi")