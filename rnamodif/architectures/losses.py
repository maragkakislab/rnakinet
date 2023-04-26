def compute_loss_enforce_A(self, output, y, loss_type):
    output = torch.nn.functional.log_softmax(output, 2)

    output_size = output.size()
    signal_len = output_size[0]
    batch_size = output_size[1]

    signal_lengths = torch.tensor(signal_len).repeat(batch_size)
    label_lengths = torch.tensor([len(label) for label in y])

    pos_indicies = [i for i, label in enumerate(y) if 'X' in label]
    neg_indicies = [i for i, label in enumerate(y) if 'X' not in label]

    negative_samples = output[:,neg_indicies,:]
    negative_labels = np.array(y)[neg_indicies]
    neg_numericalized_labels = torch.tensor([self.vocab_map[char] for label in negative_labels for char in label])
    neg_loss = torch.nn.functional.ctc_loss(
        negative_samples, 
        neg_numericalized_labels, 
        signal_lengths[neg_indicies], 
        label_lengths[neg_indicies], 
        reduction='mean', 
        blank=0 #label for the N character
    )
    
    positive_samples = output[:pos_indicies,:]
    positive_labels = np.array(y)[pos_indicies]
    pos_numericalized_labels = torch.tensor([self.vocab_map[char] for label in positive_labels for char in label])
    pos_loss = torch.nn.functional.ctc_loss(
        positive_samples,
        pos_numericalized_labels,
        signal_lengths[pos_indicies],
        label_lengths[pos_indicies],
        reduction='mean',
        blank=0
    )
    self.log(f'{loss_type} neg ctc loss', neg_loss)
    self.log(f'{loss_type} pos ctc loss', pos_loss)  

    # output = torch.nn.functional.log_softmax(output, 2)
    # weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    # weights = torch.tensor([0.6, 0.0, 0.0, 0.0, 0.0, 0.4])
    
    #TODO Finish enforcing
    
    
    weights = torch.tensor([0.3, 0.7, 0.0, 0.0, 0.0, 0.0])

    # TODO dont just take pos indicies? = loss disbalance between pos and neg?
    smooth_weight = 5
    label_smoothing_loss = smooth_weight * -((output[:,pos_indicies,:] * weights.to(output.device)).mean())

    self.log(f'{loss_type} smoothing loss', label_smoothing_loss) 

    # pos_weight = 2
    # pos_loss = pos_loss * pos_weight
    return neg_loss + pos_loss + label_smoothing_loss