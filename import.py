import torch

x = torch.empty(5, 3)
print(x.numpy())

if torch.cuda.is_available():
    print('cuda is available')

## requires_grad and zero_grad
## backward() could have all the grad computed automatically
## the gradient for this tensor will be accumulated into .grad attribute

## detach() could use to stop a tensor from tracking history and furture computation

## with torch.no_grad() can be particularly helpful for model have trainable parameters

## Tensor and Function interconnected and build up an acyclic graph

## .grad_fun attribute that references a Function created the Tensor

## .backward() on a Tensor will compute the derivatives, call .backward() on a Tensor
#need specify a gradient argument that is a tensor


x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x ** 3
#z = y.mean()
#print(y.grad_fn)
#print(z.grad_fn)
#
#z.backward()
#print(x.grad)
#print(y.grad)

y.backward(torch.ones_like(y))
print(x.grad)

x.grad.data.zero_()
print(x.grad)

