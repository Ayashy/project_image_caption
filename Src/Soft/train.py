import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from encoder import Encoder
from decoder import Decoder
from datasets import *
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


class Trainer:
    ''' 
    Trainer class to handle training
    '''

    def __init__(self, dataset='flickr', source=None, checkpoint=None, taskName=None):

        # Data parameters
        if taskName is None:
            taskName = ''
        self.taskName = taskName
        self.data_name = dataset
        if source is None:
            source = 'processed_data'
        self.data_folder = os.path.join(source, self.data_name)

        # Model parameters
        self.embedding_len = 512
        self.attention_len = 512
        self.lstm_len = 512
        self.dropout = 0.5
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.start_epoch = 0
        self.epochs = 500
        self.epochs_since_improvement = 0
        self.batch_size = 32
        self.decoder_lr = 3e-4
        self.grad_clip = 5.
        self.alpha_c = 1.
        self.best_bleu4 = 0.
        self.print_freq = 100
        self.checkpoint = checkpoint
        self.word_map = None
        self.decoder = None
        self.decoder_optimizer = None

    def start_training(self):
        """
        Training and validation.
        """

        # Loading the wordmap
        word_map_file = os.path.join(self.data_folder, 'WORDMAP.json')
        with open(word_map_file, 'r') as j:
            self.word_map = json.load(j)

        # If checkpoint exists load it, otherwise init
        if self.checkpoint is None:
            self.decoder = Decoder(attention_len=self.attention_len,
                                   embedding_len=self.embedding_len,
                                   features_len=self.lstm_len,
                                   wordmap_len=len(self.word_map))
            self.decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                                      lr=self.decoder_lr)
        else:
            self.checkpoint = torch.load(self.checkpoint)
            self.start_epoch = self.checkpoint['epoch'] + 1
            self.epochs_since_improvement = self.checkpoint['epochs_since_improvement']-1
            self.best_bleu4 = self.checkpoint['bleu-4']
            self.decoder = self.checkpoint['decoder']
            self.decoder_optimizer = self.checkpoint['decoder_optimizer']

        self.decoder = self.decoder.to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)

        # Choose the dataloader for the dataset
        if self.data_name.lower() == 'flickr':
            ds = FlickrDataset
        elif self.data_name.lower() == 'coco':
            ds = CocoDataset
        train_loader = torch.utils.data.DataLoader(
            ds(self.data_folder, 'TRAIN'),
            batch_size=self.batch_size, )
        val_loader = torch.utils.data.DataLoader(
            ds(self.data_folder, 'VAL'),
            batch_size=self.batch_size, )

        # Epochs
        for epoch in range(self.start_epoch, self.epochs):

            # If no improvement 20 times stop the learning
            if self.epochs_since_improvement == 40:
                break
            # If no improvement 8 times decay the learning_rate
            if self.epochs_since_improvement > 0 and self.epochs_since_improvement % 8 == 0:
                self.adjust_learning_rate(self.decoder_optimizer, 0.1)
            
            # Run training iteration
            self.train(train_loader=train_loader,
                       decoder=self.decoder,
                       criterion=criterion,
                       decoder_optimizer=self.decoder_optimizer,
                       epoch=epoch)

            # Run validation iteration
            recent_bleu4 = self.validate(val_loader=val_loader,
                                         decoder=self.decoder,
                                         criterion=criterion)

            # Check if there was an improvement
            is_best = recent_bleu4 > self.best_bleu4
            self.best_bleu4 = max(recent_bleu4, self.best_bleu4)
            if not is_best:
                self.epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" %
                      (self.epochs_since_improvement,))
            else:
                self.epochs_since_improvement = 0

            # Save checkpoint after each epoch
            self.save_checkpoint(self.data_name, epoch, self.epochs_since_improvement, self.decoder,
                                 self.decoder_optimizer, recent_bleu4, is_best)

    def train(self, train_loader, decoder, criterion, decoder_optimizer, epoch):
        """
        Performs one epoch's training.

        :param train_loader: DataLoader for training data
        :param decoder: decoder model
        :param criterion: loss layer
        :param decoder_optimizer: optimizer to update decoder's weights
        :param epoch: epoch number
        """

        decoder.train()  # train mode (dropout and batchnorm is used)

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy

        start = time.time()

        # Batches
        for i, (imgs, caps, caplens) in enumerate(train_loader):

            data_time.update(time.time() - start)

            # Move to GPU, if available
            caps = caps.to(device=self.device, dtype=torch.int64)
            imgs, caplens = imgs.to(self.device), caplens.to(self.device)

            # Forward prop.
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, caplens,forcing=0.5)
            
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores, _ = pack_padded_sequence(
                scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(
                targets, decode_lengths, batch_first=True)


            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            loss.backward()

            # Update weights
            decoder_optimizer.step()

            # Keep track of metrics
            top5 = self.accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                              batch_time=batch_time,
                                                                              data_time=data_time, loss=losses,
                                                                              top5=top5accs))
        
        print(scores,targets)

        

    def validate(self, val_loader, decoder, criterion):
        """
        Performs one epoch's validation.

        :param val_loader: DataLoader for validation data.
        :param encoder: encoder model
        :param decoder: decoder model
        :param criterion: loss layer
        :return: BLEU-4 score
        """
        decoder.eval()  # eval mode (no dropout or batchnorm)

        batch_time = AverageMeter()
        losses = AverageMeter()
        top5accs = AverageMeter()

        start = time.time()

        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            caps = caps.to(device=self.device, dtype=torch.int64)
            imgs, caplens = imgs.to(self.device), caplens.to(self.device)

            # Forward prop.
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, caplens,forcing=0)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(
                scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(
                targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = self.accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % self.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # because images were sorted in the decoder
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        smoothie = SmoothingFunction().method4

        bleu1 = corpus_bleu(references, hypotheses,
                            smoothing_function=smoothie, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(
            references, hypotheses, smoothing_function=smoothie, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(
            references, hypotheses, smoothing_function=smoothie, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothie, weights=(
            0.25, 0.25, 0.25, 0.25))

        print('* LOSS - {loss:.3f} \n* TOP-5 ACCURACY - {top5:.3f}\n* BLEU-1 - {bleu1} \n* BLEU-2 - {bleu2} \n* BLEU-3 - {bleu3} \n* BLEU-4 - {bleu4}\n'.format(
            loss=losses.avg,
            top5=top5accs.avg,
            bleu1=bleu1,
            bleu2=bleu2,
            bleu3=bleu3,
            bleu4=bleu4))
        return bleu4

    def adjust_learning_rate(self, optimizer, shrink_factor):
        """
        Shrinks learning rate by a specified factor.

        :param optimizer: optimizer whose learning rate must be shrunk.
        :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
        """

        print("\nDECAYING learning rate.")
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * shrink_factor
        print("The new learning rate is %f\n" %
              (optimizer.param_groups[0]['lr'],))

    def save_checkpoint(self, data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer,
                        bleu4, is_best):
        """
        Saves model checkpoint.

        :param data_name: base name of processed dataset
        :param epoch: epoch number
        :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
        :param encoder: encoder model
        :param decoder: decoder model
        :param decoder_optimizer: optimizer to update decoder's weights
        :param bleu4: validation BLEU-4 score for this epoch
        :param is_best: is this checkpoint the best so far?
        """
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'bleu-4': bleu4,
                 'decoder': decoder,
                 'decoder_optimizer': decoder_optimizer}
        filename = self.taskName+'_checkpoint_' + data_name + '.pth.tar'
        torch.save(state, os.path.join('checkpoints', filename))
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            torch.save(state, os.path.join('checkpoints', 'BEST_' + filename))

    def clip_gradient(self, optimizer, grad_clip):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.

        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def accuracy(self, scores, targets, k):
        """
        Computes top-k accuracy, from predicted and true labels.

        :param scores: scores from the model
        :param targets: true labels
        :param k: k in top-k accuracy
        :return: top-k accuracy
        """

        batch_size = targets.size(0)
        _, ind = scores.topk(k, 1, True, True)
        correct = ind.eq(targets.view(-1, 1).expand_as(ind))
        correct_total = correct.view(-1).float().sum()  # 0D tensor
        return correct_total.item() * (100.0 / batch_size)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    pass
