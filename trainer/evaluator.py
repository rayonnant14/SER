from sklearn.metrics import classification_report


class EvaluatorClassification:
    def predict(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = self.model.forward(images)
        return out

    @torch.no_grad()
    def evaluate_final_model(self):
        self.model.load_state_dict(
            torch.load(self.save_path + self.get_name() + ".pth")
        )
        self.eval_mode_on()
        outputs = [
            torch.argmax(self.predict(batch), dim=1).cpu().detach().numpy()
            for batch in self.val_dataloader
        ]
        outputs = np.concatenate(outputs, axis=0)
        targets = [
            batch[1].cpu().detach().numpy() for batch in self.val_dataloader
        ]
        targets = np.concatenate(targets, axis=0)
        print(classification_report(targets, outputs))
