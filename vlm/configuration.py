from transformers import PretrainedConfig

class phiconfig(PRetrainedConfig):
  modeltype = 'phi'
  keystooignoreatinference = ['past_key_values']

  def __init__(
    self,
    vocabsize=51200,
    hidden_size = 2048,
    intermediate_size= 8192,
    num_hidden_layers = 24,
    num_attention_heads =32,
    num_key_value_heads = None,
    
    res_drop = 0.0, #residual block droupout 퍼센트
    emb_drop=0.0,
    attention_drop = 0.0,
    
    hiddden_act = 'gelu_new',
    max_position_embedding = 2048,
    initializer_range = 0.02,
    layer_norm_eps  = 1e-5, # x-mean/std+eps
    use_cache = True,
    tie_word_embedding = False,
    
    rope_theta = 10000.0, #rotary positional encoding
    rope_scaling = None,
    partial_rotary_factor=0.5,
    qk_layernorm = False,
    bos_token_id =1, #시작 토큰
    end_token_id = 2,
    **kwargs #그외 다른변수
  ):
    self.vocabsize = vocabsize
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    
    if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

    self.num_key_value_heads = num_key_value_heads
    self.res_drop = res_drop
    self.embd_pdrop = emb_drop
    self.attention_dropout = attention_drop
    self.hidden_act = hidden_act
    self.max_position_embedding = max_position_embedding
    self.initializer_range = initializer_range
    self.layer_norm_eps = layer_norm_eps
    self.use_cache = use_cache
    self.rope_theta = rope_theta
    self.rope_scaling = rope_scaling
    self.partial_rotary_factor = partial_rotary_factor
    self.qk_layernorm = qk_layernorm
    self._rope_scaling_validation() #밑의 함수 정의함

class config(PretrainedConfig):
    model_type = "vlm"

    def __init__(self, **kwargs):
        self.phi_config = phiConfig(**kwargs)
        super().__init__(**kwargs)
