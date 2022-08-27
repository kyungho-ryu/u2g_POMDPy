import gc
class test () :
    def __init__(self):
        self.x = 5

# me 프로퍼티에 자기 자신을 할당합니다.
class RefExam():
  def __init__(self):
    print('create object')
    self.me = self
  def __del__(self):
    print(f'destroy {id(self)}')

a = RefExam()
a = 0
test = gc.collect()
print(test)
print('end .....')