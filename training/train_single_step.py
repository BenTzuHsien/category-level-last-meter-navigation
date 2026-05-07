import os, torch, numpy
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from Object_Centric_Local_Navigation.training.utils import get_least_used_gpu, get_top_available_gpus, plot_graph
from Object_Centric_Local_Navigation.training.single_step_dataset import SingleStepDataset

@torch.no_grad()
def evaluation(model, dataloader, device):
    model.eval()
    num_correct, num_total = 0, 0
    
    for current_box, current_embedding, goal_box, goal_embedding, action, prompt in tqdm(dataloader, desc="Validating", leave=False):

        current_box = current_box.to(device)
        current_embedding = current_embedding.to(device)
        goal_box = goal_box.to(device)
        goal_embedding = goal_embedding.to(device)
        action = action.to(device)

        output, _, _ = model((current_box, current_embedding), (goal_box, goal_embedding), prompt)
        prediction = torch.argmax(output, dim=2)

        num_correct += torch.all(prediction == action, dim=1).sum().item()
        num_total += action.shape[0]

    accuracy = (num_correct / num_total) * 100
    return accuracy

def train_single_step(model, dataset_paths, evaluation_path, result_path, num_gpus=1, start_index=1):
    PARAM = {
        'Batch_Size': 32,
        'Learning_Rate': 1e-4,
        'Num_Epochs': 1000,
        'Weight_Saving_Step': 10
    }

    # Tracking Parameters
    training_losses = []
    training_accuracies = []
    evaluation_accuracies = []

    # Setup Saving Path
    os.makedirs(result_path, exist_ok=True)
    weight_save_dir = os.path.join(result_path, 'weights')
    os.makedirs(weight_save_dir, exist_ok=True)
    scheduler_save_dir = os.path.join(result_path, 'schedulers')
    os.makedirs(scheduler_save_dir, exist_ok=True)
    optimizer_save_dir = os.path.join(result_path, 'optimizers')
    os.makedirs(optimizer_save_dir, exist_ok=True)
    training_losses_path = os.path.join(result_path, 'training_losses.npy')
    training_accuracies_path = os.path.join(result_path, 'training_accuracies.npy')
    evaluation_accuracies_path = os.path.join(result_path, 'evaluation_accuracies.npy')

    # Setup training dataset
    print('Loading Training Dataset...')
    if type(dataset_paths) is str:
        train_dataset = SingleStepDataset(dataset_paths)
    else:
        datasets = []
        for ds_path in dataset_paths:
            ds = SingleStepDataset(ds_path)
            datasets.append(ds)
        train_dataset = ConcatDataset(datasets)
    train_dataloader = DataLoader(train_dataset, batch_size=PARAM['Batch_Size'], shuffle=True, num_workers=16, pin_memory=True)

    # Setup evaluation dataset
    print('Loading Evaluation Dataset...')
    evaluation_dataset = SingleStepDataset(evaluation_path)
    evaluation_dataloader = DataLoader(evaluation_dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)

    # Setup model and device
    if num_gpus > 1:
        print('Multi GPU version still has some problem!!')
        return
        top_gpus = get_top_available_gpus(num_gpus)
        primary_device = f'cuda:{top_gpus[0]}'
        model = model.to(primary_device)
        model.set_goal(goal_images, prompt)
        model = torch.nn.DataParallel(model, device_ids=top_gpus)
        DEVICE = primary_device
    else:
        least_used_gpu = get_least_used_gpu()
        DEVICE = f'cuda:{least_used_gpu}'
        model = model.to(DEVICE)
    model.train()

    # Setup loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAM['Learning_Rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5, 
        patience=30,
        threshold=1e-2,
        min_lr=1e-7
    )

    # Resume previous training
    if start_index > 1:
        weight_path = os.path.join(result_path, 'weights', f'{start_index}.pth')
        model.load_weights(weight_path)
        print('Weight Loaded!')

        scheduler_path = os.path.join(result_path, 'schedulers', f'{start_index}.pth')
        scheduler.load_state_dict(torch.load(scheduler_path))
        print('Scheduler Loaded!')

        optimizer_path = os.path.join(result_path, 'optimizers', f'{start_index}.pth')
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))
        print('Optimizer Loaded!')

        training_losses = list(numpy.load(training_losses_path))[:start_index-1]
        training_losses_path = os.path.join(result_path, 'new_training_losses.npy')
        training_accuracies = list(numpy.load(training_accuracies_path))[:int(start_index/PARAM['Weight_Saving_Step'] - 1)]
        training_accuracies_path = os.path.join(result_path, 'new_training_accuracies.npy')
        evaluation_accuracies = list(numpy.load(evaluation_accuracies_path))[:int(start_index/PARAM['Weight_Saving_Step'] - 1)]
        evaluation_accuracies_path = os.path.join(result_path, 'new_evaluation_accuracies.npy')
        print('Tracking Parameter Loaded!')
    start_index -= 1

    # Training
    training_bar = tqdm(range(start_index, PARAM['Num_Epochs']), desc=f'Training {model.__class__.__name__}, Epochs')
    for epoch in training_bar:

        running_loss = torch.zeros((), device=DEVICE)
        num_correct = torch.zeros((), device=DEVICE, dtype=torch.long)
        num_total = 0
        for current_box, current_embedding, goal_box, goal_embedding, action, prompt in tqdm(train_dataloader, desc="Training", leave=False):

            current_box = current_box.to(DEVICE)
            current_embedding = current_embedding.to(DEVICE)
            goal_box = goal_box.to(DEVICE)
            goal_embedding = goal_embedding.to(DEVICE)
            action = action.to(DEVICE)

            optimizer.zero_grad()
            output, _, _ = model((current_box, current_embedding), (goal_box, goal_embedding), prompt)
            output = output.permute(0, 2, 1)   # To accomadate how CrossEnropyLoss function accept as input (Batch_size, Num_classes, ...)
            loss = loss_fn(output, action)

            # Calculate Loss
            loss.backward()
            optimizer.step()
            running_loss += loss.detach()

            # Calculate Accuracy
            prediction = torch.argmax(output, dim=1)
            num_correct += torch.all(prediction == action, dim=1).sum()
            num_total += action.shape[0]

        # Calculate Loss
        training_loss = (running_loss / len(train_dataloader)).item()
        training_losses.append([epoch+1, training_loss])
        
        # Calculate Accuracy
        train_acc = (num_correct.item() / num_total) * 100
        scheduler.step(train_acc)

        training_bar.set_description(f'Training {model.__class__.__name__}, loss={training_loss:.4f}, accuracy={train_acc:.2f}, Epochs')

        # Checkpoint
        if ((epoch + 1) % PARAM['Weight_Saving_Step']) == 0:

            training_accuracies.append([epoch+1, train_acc])
            tqdm.write(f'Training Accuracy: {train_acc}')

            eval_acc = evaluation(model, evaluation_dataloader, DEVICE)
            tqdm.write(f'Evaluation Accuracy: {eval_acc}')
            evaluation_accuracies.append([epoch+1, eval_acc])

            # Save Weight
            weight_save_path = os.path.join(weight_save_dir, f'{epoch+1}.pth')
            torch.save(model.state_dict(), weight_save_path)
            tqdm.write(f'Save Weight {epoch+1}!')

            # Save Scheduler
            scheduler_save_path = os.path.join(scheduler_save_dir, f'{epoch+1}.pth')
            torch.save(scheduler.state_dict(), scheduler_save_path)
            tqdm.write(f'Save Scheduler {epoch+1}!')

            # Save Optimizer
            optimizer_save_path = os.path.join(optimizer_save_dir, f'{epoch+1}.pth')
            torch.save(optimizer.state_dict(), optimizer_save_path)
            tqdm.write(f'Save Optimizer {epoch+1}!')

            # Save parameters
            numpy.save(training_losses_path, training_losses)
            numpy.save(training_accuracies_path, training_accuracies)
            numpy.save(evaluation_accuracies_path, evaluation_accuracies)
            model.train()

    print('Finished Training !')
    
    if PARAM['Num_Epochs'] % PARAM['Weight_Saving_Step'] != 0:

        training_accuracies.append([PARAM['Num_Epochs'], train_acc])
        tqdm.write(f'Training Accuracy: {train_acc}')

        eval_acc = evaluation(model, evaluation_dataloader, DEVICE)
        print(f'Evaluation Accuracy: {eval_acc}')
        evaluation_accuracies.append([PARAM['Num_Epochs'], eval_acc])

        # Save Weight
        weight_save_path = os.path.join(weight_save_dir, f'{epoch+1}.pth')
        torch.save(model.state_dict(), weight_save_path)
        tqdm.write(f'Save Weight {epoch+1}!')

        # Save Scheduler
        scheduler_save_path = os.path.join(scheduler_save_dir, f'{epoch+1}.pth')
        torch.save(scheduler.state_dict(), scheduler_save_path)
        tqdm.write(f'Save Scheduler {epoch+1}!')

        # Save Optimizer
        optimizer_save_path = os.path.join(optimizer_save_dir, f'{epoch+1}.pth')
        torch.save(optimizer.state_dict(), optimizer_save_path)
        tqdm.write(f'Save Optimizer {epoch+1}!')

        # Save parameters
        numpy.save(training_losses_path, training_losses)
        numpy.save(training_accuracies_path, training_accuracies)
        numpy.save(evaluation_accuracies_path, evaluation_accuracies)

    # Plot and save graphs
    plot_graph(training_losses, training_accuracies, evaluation_accuracies, PARAM, end_plot=PARAM["Num_Epochs"], figure_path=result_path)

if __name__ == '__main__':

    model_name = ''
    map_path = ''
    evaluation_path = ''
    result_path = ''

    import re, importlib
    
    module_script_name = re.sub(r'(?<!^)(?=[A-Z])', '_', model_name).lower()
    module_path = f'Object_Centric_Local_Navigation.models.{module_script_name}'
    module = importlib.import_module(module_path)
    model = getattr(module, model_name)(use_embeddings=True)

    train_single_step(model, map_path, evaluation_path, result_path)