
for t in range(epochs):
    #with torch.autograd.detect_anomaly():
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        print("Epoch1 training complete")
        torch.save(model, f"models\{model}")
        test(test_dataloader, model, loss_fn)
        print("testing complete")
model.to("cpu")
model.to(device)
print("Done!")