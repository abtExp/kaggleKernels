import numpy as np

def fit(self, x_train, y_train, batch_size):
	# NB Model Fitting
	print(np.shape(x_train), np.shape(y_train))
	num_batches = int(len(x_train)/(batch_size*10))
	for epoch in range(num_batches):
		indices = np.random.randint(0, np.shape(y_train)[0], batch_size)
		examples = [x_train[i] for i in indices]
		labels = [y_train[j] for j in indices]
		predictions = []

		for i in examples:
			#                   P(words|c=1)*P(c=1) + 1
			# P(c=1|words) =   ------------------------
			#                       P(words) + 2
			#
			#                   P(words|c=0)*P(c=0) + 1
			# P(c=0|words) =   ------------------------
			#                       P(words) + 2
			#
			# Where P(words|c=i) = product(P(word[j]|c=i))
			#
			# And P(words) = product(P(word[j]))
			#
			# 1 and 2 are added to numerator and denominator acc to laplace smoothing,
			# to avoid division by zero or numerator being 0
			p_words = np.prod(
				[word_dict[j]/np.sum(list(word_dict.values())) for j in i.split()])
			p_words += 2
			sincere_prob_num = np.prod(
				[sincere_counts[j]/self.sincere_word_count for j in i.split()]) * self.sincere_prob
			insincere_prob_num = np.prod(
				[insincere_counts[j]/self.insincere_word_count for j in i.split()]) * self.insincere_prob

			sincere_prob = sincere_prob_num/p_words
			insincere_prob = insincere_prob_num/p_words

			print('Sincere_prob: {}, Insincere_prob: {}'.format(
				sincere_prob, insincere_prob))
			predictions.append(np.argmax([sincere_prob, insincere_prob]))

		f1 = f1_score(labels, predictions)
		print('epoch {}/{}, f1_score : {}'.format(epoch, num_batches, f1))

def evaluate(self, x_val, y_val):
	predictions = []
	for example in x_val:
		p_words = np.prod(
			[word_dict[j]/np.sum(list(word_dict.values())) for j in example.split()])
		p_words += 2
		sincere_prob_num = np.prod(
			[sincere_counts[j]/self.sincere_word_count for j in example.split()]) * self.sincere_prob + 1
		insincere_prob_num = np.prod(
			[insincere_counts[j]/self.insincere_word_count for j in example.split()]) * self.insincere_prob + 1

		sincere_prob = sincere_prob_num/p_words
		insincere_prob = insincere_prob_num/p_words

		predictions.append(np.argmax([pos_prob, neg_prob]))

	f1 = f1_score(y_val, predictions)
	print(f1)

sincere_counts = Counter()
insincere_counts = Counter()
word_dict = Counter()
sincere_to_insincere_ratio = Counter()

def prepare_dicts():
    qs = [clean(i) for i in train_set['question_text']]
    lbl = [j for j in train_set['target']]
    for i,j in zip(qs,lbl):
        words = i.split()
        # making the dictionaries
        for word in words:
            word_dict[word] += 1
            if j == 0:
                sincere_counts[word] += 1
            elif j == 1:
                insincere_counts[word] += 1
    
    tst_qs = [clean(i) for i in test_set['question_text']]
    
    for i in tst_qs:
        i = i.split()
        for j in i:
            word_dict[j] += 1
    
    print('Words in sincere Questions: {}'.format(len(sincere_counts)))
    print('Words in insincere Questions: {}'.format(len(insincere_counts)))
    print('Total Words in corpus: {}'.format(len(word_dict)))

    print('Most Common Words in Sincere Questions : ')
    print(sincere_counts.most_common()[:10])
    print('Most Common Words in Insincere Questions : ')
    print(insincere_counts.most_common()[:10])

    for i in sincere_counts:
        if sincere_counts[i] >= 100:
            sincere_to_insincere_ratio[i] = np.log(sincere_counts[i]/(insincere_counts[i] + 1))

    print('The Most Sincere Words : ')
    print(sincere_to_insincere_ratio.most_common()[:10])
    print('The Most Insincere Words : ')
    print(list(reversed(sincere_to_insincere_ratio.most_common()))[:10])
    
    return sincere_counts, insincere_counts, word_dict

train_set, test_set = [], []
