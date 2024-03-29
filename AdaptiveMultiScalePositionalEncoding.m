classdef AdaptiveMultiScalePositionalEncoding
    properties
        d_model
        max_len
        alpha
        pe_coarse
        pe_detail
    end

    methods
        function obj = AdaptiveMultiScalePositionalEncoding(d_model, max_len)
            if nargin > 0
                obj.d_model = d_model;
                obj.max_len = max_len;
                obj.alpha = 0;
                
                position = (0:(max_len-1))';
                div_term = exp((0:2:(d_model-1)) * -(log(10000.0) / d_model));
                
                pe_coarse = zeros(max_len, d_model);
                pe_coarse(:, 1:2:end) = sin(position * div_term * 10);
                pe_coarse(:, 2:2:end) = cos(position * div_term * 10);
                obj.pe_coarse = pe_coarse;
                
                pe_detail = zeros(max_len, d_model);
                pe_detail(:, 1:2:end) = sin(position * div_term);
                pe_detail(:, 2:2:end) = cos(position * div_term);
                obj.pe_detail = pe_detail;
            else
                error('Not enough input arguments.');
            end
        end
        
        function output = encode(obj, seq_len, detail_level)
            if nargin < 3
                detail_level = 0.5;
            end
            
            alpha = sigmoid(obj.alpha);
            adaptive_pe = alpha * obj.pe_coarse(1:seq_len, :) + ...
                (1 - alpha) * detail_level * obj.pe_detail(1:seq_len, :);
            
            output = adaptive_pe;
        end
    end
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end
