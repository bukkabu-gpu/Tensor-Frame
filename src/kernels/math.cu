// Mathematical operations kernels
extern "C" {

// Exponential function
__global__ void exp_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expf(input[idx]);
    }
}

// Natural logarithm
__global__ void log_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = logf(input[idx]);
    }
}

// Square root
__global__ void sqrt_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sqrtf(input[idx]);
    }
}

// Power function (input^power)
__global__ void pow_kernel(const float* input, float* output, float power, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = powf(input[idx], power);
    }
}

// Sine function
__global__ void sin_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sinf(input[idx]);
    }
}

// Cosine function
__global__ void cos_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = cosf(input[idx]);
    }
}

// sinh function
__global__ void sinh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sinhf(input[idx]);
    }
}

// cosh function
__global__ void cosh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = coshf(input[idx]);
    }
}

// ReLU activation function
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU backward function
__global__ void mask_for_grad_relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (float)(input[idx]> 0.0f);
    }
}

// Sigmoid activation function
__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// Tanh activation function
__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}


__global__ void clamp_max_kernel(const float* input, float* output, float max, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx],max);
    }
}

__global__ void clamp_min_kernel(const float* input, float* output, float min, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fminf(input[idx], min);
    }
}


} // extern "C"