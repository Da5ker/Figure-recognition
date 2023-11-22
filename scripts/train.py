import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn_categorical: torch.nn.Module,
               loss_fn_numeric: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: str):
    
    model.train()
    train_loss, train_acc_cat = 0, 0

    for batch, (X, y_shape, y_color, y_size, y_angle, y_xcoord, y_ycoord) in (pbar := tqdm(enumerate(dataloader))):
        X, y_shape, y_color, y_size, y_angle, y_xcoord, y_ycoord = X.to(device), y_shape.to(device), y_color.to(device), y_size.to(device), y_angle.to(device), y_xcoord.to(device), y_ycoord.to(device)
        y_shape_pred, y_color_pred, y_size_pred, y_angle_pred, y_xcoord_pred, y_ycoord_pred = model(X)
        y_size_pred, y_angle_pred, y_xcoord_pred, y_ycoord_pred = y_size_pred.flatten(), y_angle_pred.flatten(), y_xcoord_pred.flatten(), y_ycoord_pred.flatten()
        loss_cat = loss_fn_categorical(y_shape_pred, y_shape) + loss_fn_categorical(y_color_pred, y_color)
        loss_num = loss_fn_numeric(y_size_pred, y_size) + loss_fn_numeric(y_angle_pred, y_angle) + loss_fn_numeric(y_xcoord_pred, y_xcoord) + loss_fn_numeric(y_ycoord_pred, y_ycoord)
        loss = loss_cat + loss_num
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        shape_label = y_shape_pred.argmax(dim=1)
        color_label = y_color_pred.argmax(dim=1)
        train_acc_cat += ((shape_label == y_shape) & (color_label == y_color)).sum().item()/len(shape_label)
        pbar.set_description('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch * len(X), len(dataloader.dataset),
                100. * batch / len(dataloader), loss.item()))
        
    train_loss = train_loss / len(dataloader)
    train_acc_cat = train_acc_cat / len(dataloader)

    return train_loss, train_acc_cat

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn_categorical: torch.nn.Module,
              loss_fn_numeric: torch.nn.Module,
              device: str):

    model.eval() 
    test_loss, test_acc_cat, correct_cat = 0, 0, 0
    
    with torch.inference_mode():
        for batch, (X, y_shape, y_color, y_size, y_angle, y_xcoord, y_ycoord) in enumerate(dataloader):
            X, y_shape, y_color, y_size, y_angle, y_xcoord, y_ycoord = X.to(device), y_shape.to(device), y_color.to(device), y_size.to(device), y_angle.to(device), y_xcoord.to(device), y_ycoord.to(device)
            y_shape_test, y_color_test, y_size_test, y_angle_test, y_xcoord_test, y_ycoord_test = model(X)
            y_size_test, y_angle_test, y_xcoord_test, y_ycoord_test = y_size_test.flatten(), y_angle_test.flatten(), y_xcoord_test.flatten(), y_ycoord_test.flatten()
            loss_cat = loss_fn_categorical(y_shape_test, y_shape) + loss_fn_categorical(y_color_test, y_color)
            loss_num = loss_fn_numeric(y_size_test, y_size) + loss_fn_numeric(y_angle_test, y_angle) + loss_fn_numeric(y_xcoord_test, y_xcoord) + loss_fn_numeric(y_ycoord_test, y_ycoord)
            loss = loss_cat + loss_num
            test_loss += loss.item()
            shape_label = y_shape_test.argmax(dim=1)
            color_label = y_color_test.argmax(dim=1)     
            test_acc_cat += ((shape_label == y_shape) & (color_label == y_color)).sum().item()/len(shape_label)
            correct_cat += ((shape_label == y_shape) & (color_label == y_color)).sum().item()  

    test_loss = test_loss / len(dataloader)
    test_acc_cat = test_acc_cat / len(dataloader)
    
    return test_loss, test_acc_cat, correct_cat

def epohs(model: torch.nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          test_loader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn_categorical: torch.nn.Module = nn.CrossEntropyLoss(),
          loss_fn_numeric: torch.nn.Module = nn.MSELoss(),
          epochs: int = 5,
          path='C:\\Users\\User\\AppData\\Local\\Programs\\VS_Projects\\Testing\\Figure project\\files'):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {"train_loss": [],
        "train_acc_cat": [],
        "test_loss": [],
        "test_acc_cat": []}
    
    for epoch in range(epochs):
        train_loss, train_acc_cat = train_step(model=model,
                                            dataloader=train_loader,
                                            loss_fn_categorical=loss_fn_categorical,
                                            loss_fn_numeric=loss_fn_numeric,
                                            optimizer=optimizer,
                                            device=device)
        test_loss, test_acc_cat, correct_cat = test_step(model=model,
                                                    dataloader=test_loader,
                                                    loss_fn_categorical=loss_fn_categorical,
                                                    loss_fn_numeric=loss_fn_numeric,
                                                    device=device)
        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc_cat: {train_acc_cat:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc_cat: {test_acc_cat:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc_cat"].append(train_acc_cat)
        results["test_loss"].append(test_loss)
        results["test_acc_cat"].append(test_acc_cat)
    
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    correct_cat, len(test_loader.dataset),
    100. * correct_cat / len(test_loader.dataset)))

    torch.save(model.state_dict(), path + '\\model.pth')
    torch.save(optimizer.state_dict(), path + '\\optimizer.pth')

    return results

def plot_loss_curves(results):

    loss = results['train_loss']
    test_loss = results['test_loss']
    accuracy = results['train_acc_cat']
    test_accuracy = results['test_acc_cat']
    epochs = range(1, len(results['train_loss'])+1)

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy_cat')
    plt.plot(epochs, test_accuracy, label='test_accuracy_cat')
    plt.title('Accuracy_cat')
    plt.xlabel('Epochs')
    plt.legend()