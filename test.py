import torch

# 텐서 공장 메소드에 ``requires_grad`` 플래그가 있습니다.
x = torch.tensor([1., 2., 3], requires_grad=True)

# requires_grad=True 으로 이전에 있었던 모든 작업을 여전히 할 수 있습니다.
y = torch.tensor([4., 5., 6], requires_grad=True)
z = x + y
print(z)

# 그러나 z 는 몇가지를 추가로 알고 있습니다.
print(z.grad_fn)