class PatchLayer(Layer):
	"""
	Layering and transforming images into patches.
	"""
	def __init__(self, img_size, patch_size, num_patches, projection_dim, **kwargs):
		super(PatchLayer, self).__init__(**kwargs)
		self.img_size = img_size
		self.patch_size = patch_size
		self.num_patches = num_patches
		self.projection_dim = projection_dim
		self.half_patch = patch_size // 2
		self.flatten_patches = layers.Reshape((num_patches, -1))
		self.projection = layers.Dense(units=projection_dim)
		self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

	def shiftImg(self, images, mode):
		# Building diagonally-shifted images
		if mode == 'left-up':
			cropheight = self.half_patch
			cropwidth = self.half_patch
			shiftheight = 0
			shiftwidth = 0
		elif mode == 'left-down':
			cropheight = 0
			cropwidth = self.half_patch
			shiftheight = self.half_patch
			shiftwidth = 0
		elif mode == 'right-up':
			cropheight = self.half_patch
			cropwidth = 0
			shiftheight = 0
			shiftwidth = self.half_patch
		else:
			cropheight = 0
			cropwidth = 0
			shiftheight = self.half_patch
			shiftwidth = self.half_patch

		crop = tf.image.crop_to_bounding_box(
			images,
			offset_height=cropheight,
			offset_width=cropwidth,
			target_height=self.img_size - self.half_patch,
			target_width=self.img_size - self.half_patch
		)

		shiftPad = tf.image.pad_to_bounding_box(
			crop,
			offset_height=shiftheight,
			offset_width=shiftwidth,
			target_height=self.img_size,
			target_width=self.img_size
		)
		return shiftPad

	def call(self, images):
		images = tf.concat(
			[
				images,
				self.shiftImg(images, mode='left-up'),
				self.shiftImg(images, mode='left-down'),
				self.shiftImg(images, mode='right-up'),
				self.shiftImg(images, mode='right-down'),
			],
			axis=-1
		)
		patches = tf.image.extract_patches(
			images=images,
			sizes=[1, self.patch_size, self.patch_size, 1],
			strides=[1, self.patch_size, self.patch_size, 1],
			rates=[1, 1, 1, 1],
			padding='VALID'
		)
		flat_patches = self.flatten_patches(patches)
		tokens = self.layer_norm(flat_patches)
		tokens = self.projection(tokens)

		return (tokens, patches)

	def getConfig_(self):
		config_ = super(PatchLayer, self).getConfig_()
		config_.update(
			{
				'img_size': self.img_size,
				'patch_size': self.patch_size,
				'num_patches': self.num_patches,
				'projection_dim': self.projection_dim
			}
		)
		return config_

class Embed_Patch(Layer):
	"""
	Layering for projecting patches into a vector.
	"""
	def __init__(self, num_patches, projection_dim, **kwargs):
		super(Embed_Patch, self).__init__(**kwargs)
		self.num_patches = num_patches
		self.projection_dim = projection_dim
		self.position_embedding = layers.Embedding(
			input_dim=self.num_patches, output_dim=projection_dim
		)

	def call(self, patches):
		positions = tf.range(0, self.num_patches, delta=1)
		return patches + self.position_embedding(positions)

	def getConfig_(self):
		config_ = super(Embed_Patch, self).getConfig_()
		config_.update(
			{
				'num_patches': self.num_patches,
				'projection_dim': self.projection_dim
			}
		)
		return config_

class Multi_Head_AttentionLSA(layers.MultiHeadAttention):
	"""
	Multi Head Attention layer for the transformer-encoder block, but with the
	addition of using Local-Self-Attention to improve feature	learning.

	"""
	def __init__(self, **kwargs):
		super(Multi_Head_AttentionLSA, self).__init__(**kwargs)
		self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

	def computeAttention(self, query, key, value, attention_mask=None,
			training=None):
		query = tf.multiply(query, 1.0/self.tau)
		attention_scores = tf.einsum(self._dot_product_equation, key, query)
		attention_mask = tf.convert_to_tensor(attention_mask)
		attention_scores = self._masked_softmax(attention_scores, attention_mask)
		attention_scores_dropout = self._dropout_layer(
			attention_scores, training=training
		)
		attention_output = tf.einsum(
			self._combine_equation, attention_scores_dropout, value
		)
		return attention_output, attention_scores

	def getConfig_(self):
		config_ = super(Multi_Head_AttentionLSA, self).getConfig_()
		return config_


def buildVisionTransformer(input_shape, img_size, patch_size, num_patches,
			attention_heads, projection_dim, hidden_units, dropout_rate,
			transf_layers, mlp_head_units):
	"""
	Building the vision transformer.
	"""
	# Input layer
	inputs = layers.Input(shape=input_shape)

	# Convert image data into patches
	(tokens, _) = PatchLayer(
		img_size,
		patch_size,
		num_patches,
		projection_dim
	)(inputs)

	# Encode patches
	encodedPatches = Embed_Patch(num_patches, projection_dim)(tokens)

	# Create transformer layers
	for _ in range(transf_layers):
		# First layer normalisation
		layerNorm1 = layers.LayerNormalization(
			epsilon=1e-6
		)(encodedPatches)

		# Build diagoanl attention mask
		diagAttnMask = 1 - tf.eye(num_patches)
		diagAttnMask = tf.cast([diagAttnMask], dtype=tf.int8)

		# Multi-head attention layer
		attention_output = Multi_Head_AttentionLSA(
			num_heads=attention_heads, key_dim=projection_dim,
			dropout=dropout_rate
		)(layerNorm1, layerNorm1, attention_mask=diagAttnMask)

		# First skip connection
		skip1 = layers.Add()([attention_output, encodedPatches])

		# Second layer normalisation
		layerNorm2 = layers.LayerNormalization(epsilon=1e-6)(skip1)

		# Multi-Layer Perceptron
		mlpLayer = layerNorm2
		for units in hidden_units:
			mlpLayer = layers.Dense(units, activation=tf.nn.gelu)(mlpLayer)
			mlpLayer = layers.Dropout(dropout_rate)(mlpLayer, training=False)

		# Second skip connection
		encodedPatches = layers.Add()([mlpLayer, skip1])

	# Create a [batch_size, projection_dim] tensor
	representtn = layers.LayerNormalization(epsilon=1e-6)(encodedPatches)
	representtn = layers.Flatten()(representtn)
	representtn = layers.Dropout(dropout_rate)(representtn, training=False)

	# MLP layer for learning features
	features = representtn
	for units in mlp_head_units:
		features = layers.Dense(units, activation=tf.nn.gelu)(features)
		features = layers.Dropout(dropout_rate)(features, training=False)

	# Classify outputs
	logits = layers.Dense(1)(features)

	# Create Keras model
	model = tf.keras.Model(inputs=inputs, outputs=logits)

	return model
