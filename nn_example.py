import torch
import custom_nn as twolayer
import DynamicNet as dyn
N,D_in, H, D_out = 64,1000,100,10

x=torch.randn(N,D_in)
y=torch.randn(N,D_out)

# model = torch.nn.Sequential(torch.nn.Linear(D_in,H),
#         torch.nn.ReLU(),
#         torch.nn.Linear(H,D_out),)

# loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4

# model = twolayer.TwoLayerNet(D_in,H,D_out)
model = dyn.DynamicNet(D_in,H,D_out)
criterion = torch.nn.MSELoss(size_average=False)

# create optimizer

# optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
for t in range(500):

    y_pred = model(x)

    # loss = loss_fn(y_pred,y)
    loss = criterion(y_pred,y)
    print(t,loss.item())

    # model.zero_grad()
    optimizer.zero_grad()

    loss.backward()

    # with torch.no_grad():
    #     for param in model.parameters():
    #         param -= learning_rate * param.grad
    optimizer.step()