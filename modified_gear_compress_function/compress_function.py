import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def fake_groupwise_token_asymmetric_quantization( ####
    input: torch.Tensor, quantize_bit, group_size=128
):
        
    print("not supposed to reach, fake_groupwise_token_asymmetric_quantization")

    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    ).float()
    num_groups = (sep_dim * num_head) // group_size
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    scale = (mx - mn) / (2**quantize_bit - 1)
    input_in_groups = (input_in_groups - mn) / scale
    input_in_groups = F.relu(input_in_groups)
    rounded_input_in_groups = input_in_groups.round_()
    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    dequantized_input = dequantized_input_in_groups.view(
        batch, seq_len, num_head, sep_dim
    )
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def fake_groupwise_channel_asymmetric_quantization_new(
    input: torch.Tensor, quantize_bit, group_size=128
):

    print("not supposed to reach, fake_groupwise_channel_asymmetric_quantization_new")

    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    # group_size = 128
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    input = input.view(batch, seq_len, num_head * sep_dim)
    group_num = input.shape[1] // group_size

    fixed_input = input.view(batch,group_num, group_size, num_head * sep_dim)
    mx, mn = fixed_input.max(dim=-2)[0], fixed_input.min(dim=-2)[0]
    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)
    
    scale = (mx - mn) / (2**quantize_bit - 1)
    quantized_input = (fixed_input - mn) / scale
    quantized_input = F.relu(quantized_input)
    rounded_input = quantized_input.round_()
    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch,group_num * group_size,num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape

    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input

def fake_poweriteration_group(input: torch.Tensor, loop, rank, device, p_base, q_base):
    # input size [batch,num_head,seq_len,model_dim/num_head]
    # -> [batch,seq_len,model_dim] -> [batch * seq_len,model_dim]
    # p_base = torch.rand(input.shape[3] * input.shape[1], rank).to(device)
    # q_base = torch.rand(input.shape[0] * input.shape[2], rank).to(device)
    
    debug = False

    if (debug):
        print("reached fake_poweriteration_group")
    
    dtype = input.dtype
    batch, dim1, dim2, dim3 = input.shape

    if (debug):
        print("batch, dim1, dim2, dim3: ", batch, dim1, dim2, dim3)
        print("loop, rank: ", loop, rank)

    input = input.float()
    
    if (debug):
        print("input:", input)

        print("q_base: ", q_base)
        print("p_base: ", p_base)

    if q_base is not None and p_base is not None:
        
        if (debug):
            print("q_base is not None, p_base is not None")
        
        p_base[0] = p_base[0].float()
        q_base[0] = q_base[0].float()
    else:
        
        if (debug):
            print("q_base or p_base is none")
        
        p_base = [torch.rand(batch,dim1,dim3, rank).to(input.device)]
        q_base = [torch.rand(batch,dim1,dim2, rank).to(input.device)]
   
    if (debug):
        print("q_base after init: ", q_base)
        print("q_base after init len: ", len(q_base))
        print("p_base after init: ", p_base)
        print("p_base after init len: ", len(p_base))

   # 3 calculation = loop * (matmul) + 2 * qrO(n^2)
    for i in range(loop):
        if i == loop - 1:
            p_base[0] = torch.linalg.qr(p_base[0]).Q
        q_base[0] = input @ p_base[0]
        if i == loop - 1:
            q_base[0] = torch.linalg.qr(q_base[0]).Q
        p_base[0] = torch.transpose(input, 2, 3) @ q_base[0]
    input = q_base[0] @ torch.transpose(p_base[0], 2, 3)
   
    if (debug):
        print("q_base: ", q_base)
        print("q_base[0]: ", q_base[0])
        print("q_base[0] type: ", type(q_base[0][0]))
        
       
    if (debug):
        print("p_base: ", p_base)
        print("p_base[0]: ", p_base[0])
        print("p_base[0] type: ", type(p_base[0][0]))
        print("p_base[0] shape: ", p_base[0].shape)
        p_transpose = torch.transpose(p_base[0], 2, 3)
        print("p_base[0] transpose shape: ", p_transpose.shape)

    input = input.view(batch, dim1, dim2, dim3)

    if (debug):
        print("input after qr decomp: ", input)
        print("input after qr decomp shape: ", input.shape)

    input = input.type(dtype)

    return input

def fake_groupwise_channel_asymmetric_quantization_cluster(input,cluster_num,group_size=128, layer_idx = 0, isKey = True):
    
    debug = False

    print("reached GEAR's quant")

    if (debug):
        print("reached fake_groupwise_channel_asymmetric_quantization_cluster")

    batch, num_head, seq_len, sep_dim = input.shape

    if (debug):
        print("batch, num_head, seq_len, sep_dim: ", batch, num_head, seq_len, sep_dim)

    dtype = input.dtype
    # group_size = 128

    if (debug):
        print("input before permute: ", input)
    if (debug):
        print("input before permute shape: ", input.shape)

    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    input = input.view(batch, seq_len, num_head * sep_dim)
    
    if (debug):
        print("input after permute: ", input)
    if (debug):
        print("input after permute shape: ", input.shape)

    group_num = input.shape[1] // group_size
    fixed_length = int(group_num * group_size)
    fixed_input = input[:,:fixed_length,:]
    residual_input = input[:,fixed_length:,:]
    fixed_input = fixed_input.view(batch,group_num, group_size, num_head * sep_dim)

    debug = True

    if (debug):
        print("group_size: ", group_size)
    
    debug = False

    if (debug):
        print("group_num: ", group_num)
    if (debug):
        print("fixed_length: ", fixed_length)
    if (debug):
        print("fixed_input: ", fixed_input)
    if (debug):
        print("fixed_input shape: ", fixed_input.shape)
    if (debug):
        print("residual_input: ", residual_input)
    if (debug):
        print("residual_input shape: ", residual_input.shape)

    mx, mn = fixed_input.max(dim=-2)[0], fixed_input.min(dim=-2)[0]
    
    if (debug):
        print("mx: ", mx)
    if (debug):
        print("mx shape: ", mx.shape)
    if (debug):
        print("mn: ", mn)

    debug = True

    if (debug):
        print("mn shape: ", mn.shape)

    debug = False

    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)

    if (debug):
        print("mx after unsqueeze: ", mx)
    if (debug):
        print("mx after unsqueeze shape: ", mx.shape)
    if (debug):
        print("mn after unsqueeze: ", mn)
    
    debug = True

    if (debug):
        print("mn after unsqueeze shape: ", mn.shape)

    debug = False

    scale = (mx - mn) / cluster_num

    if (debug):
        print("cluster_num: ", cluster_num)
    if (debug):
        print("scale: ", scale)
    if (debug):
        print("scale.shape: ", scale)

    quantized_input = (fixed_input - mn) / scale

    if (debug):
        print("quantized_input: ", quantized_input)
    if (debug):
        print("quantized_input.shape: ", quantized_input.shape)

    quantized_input = F.relu(quantized_input)

    if (debug):
        print("quantized_input after relu: ", quantized_input)
    if (debug):
        print("quantized_input after rule shape: ", quantized_input.shape)

    rounded_input = quantized_input.round_()

    if (debug):
        print("rounded input: ", rounded_input)
       
    print("rounded input shape: ", rounded_input.size())

    prefix = 'k' if isKey else 'v'
    #print("test prefix: should be k. prefix: ", prefix, " layer_idx: ", layer_idx)
    #torch.save(rounded_input, f"{prefix}_{layer_idx}.pt")
    #print(f"{prefix} rounded_input size, values, max :", rounded_input.unique().size(),
    #        rounded_input.unique(),
    #        rounded_input.max())

    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch,group_num * group_size,num_head * sep_dim)
    
    if (debug):
        print("dequantized input: ", dequantized_input)
    if (debug):
        print("dequantized input shape: ", dequantized_input)

    concat_input = torch.cat((dequantized_input,residual_input),dim=1)
    
    if (debug):
        print("concat_input: ", concat_input)
    if (debug):
        print("concat_input shape: ", concat_input)

    dequantized_input = concat_input.view(batch, seq_len, num_head, sep_dim)
    
    if (debug):
        print("dequantized_input after concat: ", dequantized_input)
    if (debug):
        print("dequantized_input after shape: ", dequantized_input.shape)

    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)

    if (debug):
        print("dequantized input after permute: ", dequantized_input)

    # reshape the input back to its original shape

    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)

    if (debug):
        print("input after permute: ", input)
    if (debug):
        print("input after permute shape: ", input.shape)
    
    return dequantized_input

def fake_groupwise_token_asymmetric_quantization_cluster(input,cluster_num,group_size=128, layer_idx = 0, isKey = True):
    
    debug = False

    print("reached GEAR's quant")

    if (debug):
        print("input: ", input)
    
    batch, num_head, seq_len, sep_dim = input.shape
    
    if (debug):
        print("batch, num_head, seq_len, sep_dim: ", batch, num_head, seq_len, sep_dim)

    dtype = input.dtype
    
    if (debug):
        print("input before permute: ", input)
        print("input before permute shape: ", input.shape)

    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    if (debug):
        print("input after permute: ", input)
        print("input after permute shape: ", input.shape)

    num_groups = (sep_dim * num_head) // group_size

    if (debug):
        print("group_size, num: ", group_size, num_groups)
    
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    if (debug):
        print("input_in_groups: ", input_in_groups)
        print("input_in_groups shape: ", input_in_groups.shape)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]

    if (debug):
        print("mx: ", mx)
   
    #print("mx shape: ", mx.shape)
        
    if (debug):
        print("mn: ", mn)
        print("mn shape: ", mn.shape) 

    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    if (debug):
        print("mx after unsqueeze: ", mx)
        print("mx after unsqueeze shape: ", mx.shape)
        print("mn after unsqueeze: ", mn)
        print("mn after unsqueeze shape: ", mn)

    scale = (mx - mn) / cluster_num

    if (debug):
        print("cluster_num: ", cluster_num)
        print("scale: ", scale)
        print("scale.shape: ", scale)

    input_in_groups = (input_in_groups - mn) / scale

    if (debug):
        print("input_in_groups: ", input_in_groups)
        print("input_in_groups.shape: ", input_in_groups.shape)

    input_in_groups = F.relu(input_in_groups)

    if (debug):
        print("input_in_groups after rele: ", input_in_groups)
        print("input_in_groups.shape after rule: ", input_in_groups.shape)

    rounded_input_in_groups = input_in_groups.round_()
    
    if (debug):
        print("rounded_input_in_groups: ", rounded_input_in_groups)
        print("rounded_input_in_groups shape: ", rounded_input_in_groups.size())

    prefix = 'k' if isKey else 'v'
    #print("test prefix: should be v. prefix: ", prefix)
    #torch.save(rounded_input_in_groups, f"{prefix}_{layer_idx}.pt")
    #print(f"{prefix} rounded_input_in_groups size, values, max :", rounded_input_in_groups.unique().size(),
    #        rounded_input_in_groups.unique(),
    #        rounded_input_in_groups.max())

    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    
    if (debug):
        print("dequantized_input_in_groups: ", dequantized_input_in_groups)
        print("dequantized_input_in_groups shape: ", dequantized_input_in_groups.shape)

    dequantized_input = dequantized_input_in_groups.view(
        batch, seq_len, num_head, sep_dim
    )

    if (debug):
        print("dequantized_input: ", dequantized_input)
        print("dequantized_input shape: ", dequantized_input.shape)

    dequantized_input = dequantized_input.permute(0, 2, 1, 3)

    if (debug):
        print("dequantized_input after permute: ", dequantized_input)
        print("dequantized_input after permute shape: ", dequantized_input.shape)

    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)

    #torch.save(dequantized_input, f"dequantized_{prefix}_{layer_idx}.pt")

    return dequantized_input


def gearslkivi_channelQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):

    print("not supposed to reach, gearslkivi_channelQ")

    input = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    sparsity_num = int(element_num * sparsity)
    # print(sparsity_num,sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)
    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    quantized_output = gearlkivi_channelQ(input, quantize_bit, group_size,rank,loop)
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    input.scatter_(-1, smallest_indices, smallest_value)
    input.scatter_(-1, largest_indices, largest_value)
    

    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    input = input.half()
    quantized_output = quantized_output.half()

    
    return quantized_output



def gearslkivi_tokenQ_new(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1, layer_idx = 0, isKey = True): ####
    
    use_error = False

    #print("reached gearslkivi_tokenQ_new")

    input = input.float()
    cloned_input = input.clone()
    output = gears_tokenQ(input, quantize_bit, group_size,sparsity, layer_idx, isKey)

    error = cloned_input - output


    #print("error:", error)
    toReturn = output

    if (use_error):
        error_lr = fake_poweriteration_group(error, loop, rank, input.device, None, None)

        #diff = torch.sub(output + error_lr, output)
        #torch.set_printoptions(profile="full")
        #print("diff: ", diff)
        #print("shape: ", diff.shape)

        print("using error")

        toReturn =  output + error_lr

    #prefix = 'k' if isKey else 'v'
    #print("test prefix: should be v. prefix: ", prefix, " layer_idx: ", layer_idx)
    #deltas = toReturn - input
    #torch.save(deltas, f"{prefix}_gear_{layer_idx}.pt")


    return toReturn

def gearslkivi_channelQ_new(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1, layer_idx = 0, isKey = True): ####
   
    debug = False
    use_error = False

    if (debug):
        print("reached gearslkivi_channelQ_new")
        print("input shape: ", input.shape)

    input = input.float()

    if (debug):
        print("input shape after float: ", input.shape)

    cloned_input = input.clone()

    if (debug):
        print("cloned input shape: ", cloned_input.shape)

    output = gears_channelQ(input, quantize_bit, group_size,sparsity, layer_idx, isKey)

    if (debug):
        print("output after gears_channelQ shape: ", output.shape)

    error = cloned_input - output

    if (debug):
        print("error: ", error)
        print("error shape: ", error.shape)

    toReturn = output

    if (use_error):
        error_lr = fake_poweriteration_group(error, loop, rank, input.device, None, None)
        
        #diff = torch.sub(output + error_lr, output)
        #torch.set_printoptions(profile="full")
        #print("diff: ", diff)
        #print("shape: ", diff.shape)

        print("using error")

        toReturn =  output + error_lr

    #prefix = 'k' if isKey else 'v'
    #print("test prefix: should be k. prefix: ", prefix, " layer_idx: ", layer_idx)
    #deltas = toReturn - input
    #torch.save(deltas, f"{prefix}_gear_{layer_idx}.pt")

    return toReturn

def gearslkivi_tokenQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):

    print("not supposed to reach, gearslkivi_tokenQ")

    input = input.float()

    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    input = input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)
    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3) 
    quantized_output = gearlkivi_tokenQ(input, quantize_bit, group_size,rank,loop)
    # Restore the original values at the smallest and largest k indices
    quantized_output = quantized_output = (
        quantized_output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    quantized_output.scatter_(-1, smallest_indices, smallest_value)
    quantized_output.scatter_(-1, largest_indices, largest_value)
    

    quantized_output = quantized_output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    quantized_output = quantized_output.half()
    return quantized_output

def standard_4bit_q(input, layer_idx = 0, isKey = True):
    print("reached standard 4bit q")
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype

    flattened_input = input.reshape(-1)

    mx = flattened_input.max()
    mn = flattened_input.min()

    levels = 16
    scale = (mx - mn) / (levels - 1)
    zero_point = mn

    quantized_input = torch.round((flattened_input - zero_point) / scale)

    quantized_input = torch.clamp(quantized_input, 0, levels - 1)

    dequantized_input = (quantized_input * scale) + zero_point

    dequantized_input = dequantized_input.view(batch, num_head, seq_len, sep_dim)

    dequantized_input = dequantized_input.type(dtype)

    return dequantized_input


def gears_channelQ2(input, quantize_bit, group_size=128,sparsity=0.0, layer_idx = 0, isKey = True):

    #print("not using outliers")
    standard_4bit = False

    output = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    sparsity_num = int(element_num * sparsity)
    # print(sparsity_num,sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)

    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    # Find the indices of the smallest k elements along the last dimension
    #smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    #largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)
    #average = output.mean(dim=-1, keepdim=True)
    #expanded_average = average.expand_as(output)
    #index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    # Set the smallest k elements to the average value
    #output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    #output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    

    if (not standard_4bit):
        output = fake_groupwise_channel_asymmetric_quantization_cluster(
            output, quantize_bit ** 2 - 1, group_size, layer_idx, isKey)
    else:
        output = standard_4bit_q(output, layer_idx, isKey)


    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    #output.scatter_(-1, smallest_indices, smallest_value)
    #output.scatter_(-1, largest_indices, largest_value)


    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = output.half()
    return output
   

def gears_channelQ(input, quantize_bit, group_size=128,sparsity=0.0, layer_idx = 0, isKey = True):
    
    debug = False
    no_outlier = True

    if (no_outlier):
        return gears_channelQ2(input, quantize_bit, group_size, sparsity, layer_idx, isKey)

    print("using outliers")

    #print("reached gears_channelQ")
    output = input.float()
   
    #torch.set_printoptions(profile="full")

    if (debug):
        print("output: ", output)
        print("output shape: ", output.shape)

    batch, num_head, seq_len, sep_dim = input.shape
    
    if (debug):
        print("batch, num head, seq len, sep dim: ", batch, num_head, seq_len, sep_dim)
    
    element_num = batch * num_head * seq_len * sep_dim
    sparsity_num = int(element_num * sparsity)
    
    if (debug):
        print("element, sparsity, sparsity_num: ", element_num, sparsity, sparsity_num)
    
    # print(sparsity_num,sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    
    if (debug):
        print("sparsity per token: ", sparsity_pertoken)

    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )

    if (debug):
        print("output after permute: ", output)
        print("output after permute shape: ", output.shape)
    
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)
    average = output.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(output)
    index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    
    if (debug):
        print("smallest val, smallest indices: ", smallest_value, smallest_indices)
        
    #print("smallest shapes: ", smallest_value.shape, smallest_indices.shape)
        
    if (debug):
        print("largest val, indices: ", largest_value, largest_indices)
        print("largest shapes: ", largest_value.shape, largest_indices.shape)
        print("average: ", average)
        print("expanded avg: ", expanded_average)
        print("index_helper: ", index_helper)

    # Set the smallest k elements to the average value
    output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    if (debug):
        print("output after smallest k to avg: ", output)

    # Set the largest k elements to the average value
    output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    
    if (debug):
       print("output after largest k to avg: ", output)

    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    output = fake_groupwise_channel_asymmetric_quantization_cluster(
       output, quantize_bit ** 2 - 1, group_size, layer_idx, isKey)
   
    if (debug):
        print("output after fake_groupwise_channel_asymmetric_quantization_cluster shape: ", output.shape)

    output = (
        output.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )

    if (debug):
        print("output after permute shape: ", output.shape)

    output.scatter_(-1, smallest_indices, smallest_value)
    output.scatter_(-1, largest_indices, largest_value)
   
    if (debug):
        print("output after scatter: ", output)
        print("output after scatter shape: ", output.shape)

    output = output.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    
    if (debug):
        print("output after view shape: ", output.shape)

    output = output.half()
    
    if (debug):
        print("output after half: ", output)
        
    #print("output after half shape: ", output.shape)

    return output

def gears_tokenQ2(input, quantize_bit, group_size=128,sparsity=0.0, layer_idx = 0, isKey = True):
    
    #print("not using outliers")
    standard_4bit = False

    output = input.float()
    batch, num_head, seq_len, sep_dim = output.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    # Find the indices of the smallest k elements along the last dimension
    #smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    #largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)
    #average = output.mean(dim=-1, keepdim=True)
    #expanded_average = average.expand_as(output)
    #index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    # Set the smallest k elements to the average value
    #output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    #output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    
    if (not standard_4bit):
        output = fake_groupwise_token_asymmetric_quantization_cluster(
            output, quantize_bit ** 2 - 1, group_size, isKey)
    else:
        output = standard_4bit_q(output, layer_idx, isKey)

    # Restore the original values at the smallest and largest k indices
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    #output.scatter_(-1, smallest_indices, smallest_value)
    #output.scatter_(-1, largest_indices, largest_value)
    

    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = output.half()
    return output


def gears_tokenQ(input, quantize_bit, group_size=128,sparsity=0.0, layer_idx = 0, isKey = True):
    
    debug = False
    no_outlier = True

    if (no_outlier):
        return gears_tokenQ2(input, quantize_bit, group_size, sparsity, layer_idx, isKey)

    print("using outlier")

    output = input.float()
    
    if (debug):
       print("before output: ", output)

    batch, num_head, seq_len, sep_dim = output.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(output, sparsity_pertoken, dim=-1, largest=True)
    average = output.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(output)
    index_helper = torch.arange(output.size(-1), device=output.device).expand_as(output)
    # Set the smallest k elements to the average value
    output.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    output.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = fake_groupwise_token_asymmetric_quantization_cluster(
        output, quantize_bit ** 2 - 1, group_size, layer_idx, isKey)
    
    if (debug):
        print("output after smallest and largest set to avg value: ", output)
        print("output after smallest and largest set to avg value shape: ", output.size())

    # Restore the original values at the smallest and largest k indices
    output = (
        output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    if (debug):
        print("output after original vals are restored: ", output)
        print("output after original vals are restored shape: ", output.size())

    output.scatter_(-1, smallest_indices, smallest_value)
    output.scatter_(-1, largest_indices, largest_value)
    

    output = output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    output = output.half()

    if (debug):
        print("after output: ", output)
    
    return output


def tokenwise_gearlkivi_channelQ(input, quantize_bit, group_size=128,r=0,loop=1): ####
    
    use_error = False

    bsz, num_head, seq_len, sep_dim = input.shape
    cloned_input = input.clone()
    output = fake_groupwise_channel_asymmetric_quantization_new(
        input, quantize_bit, group_size
    )
    
    error = cloned_input - output
    #### TODO some changes here
    # error = error.permute(0, 1, 3, 2).contiguous().view(bsz, sep_dim * num_head, seq_len)
    # group_num = seq_len // group_size
    # error = error.view(bsz, sep_dim * num_head, group_num, group_size)
    
    if (use_error):
        error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,

                                )
        # error_lr = error_lr.view(bsz, sep_dim, num_head, group_num*group_size).permute(0, 2, 3, 1).contiguous().view(bsz, num_head, group_num*group_size, sep_dim)
        
        print("using error")

        return output + error_lr

    return output

def gearlkivi_channelQ(input, quantize_bit, group_size=128,r=0,loop=1):
    
    use_error = False

    bsz, num_head, seq_len, sep_dim = input.shape
    output = fake_groupwise_channel_asymmetric_quantization_new(
        input, quantize_bit, group_size
    )
    
    error = input - output
    #### TODO some changes here
    # error = error.permute(0, 1, 3, 2).contiguous().view(bsz, sep_dim * num_head, seq_len)
    # group_num = seq_len // group_size
    # error = error.view(bsz, sep_dim * num_head, group_num, group_size)
    
    if (use_error):
        error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,
                                )
        # error_lr = error_lr.view(bsz, sep_dim, num_head, group_num*group_size).permute(0, 2, 3, 1).contiguous().view(bsz, num_head, group_num*group_size, sep_dim)
   
        print("using error")

        return output + error_lr

    return output

def gearlkivi_tokenQ(input, quantize_bit, group_size=128,r=0,loop=1):
    
    use_error = False

    bsz, num_head, seq_len, sep_dim = input.shape
    output = fake_groupwise_token_asymmetric_quantization(
        input, quantize_bit, group_size
    )
    error = input - output
    # error = error.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, sep_dim * num_head)
    # num_groups = (sep_dim * num_head) // group_size
    # error = error.view(bsz, seq_len, num_groups, group_size)
    
    if (use_error):
        error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None
                                )
        # error_lr = error_lr.view(bsz, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
        
        #diff = torch.sub(output + error_lr, output)
        #torch.set_printoptions(profile="full")
        #print("diff: ", diff)
        #print("shape: ", diff.shape)

        print("using error")

        return output + error_lr

    return output

def tokenwise_gearlkivi_tokenQ(input, quantize_bit, group_size=128,r=0,loop=1): ####
    
    use_error = False

    bsz, num_head, seq_len, sep_dim = input.shape
    cloned_input = input.clone()
    output = fake_groupwise_token_asymmetric_quantization(
        input, quantize_bit, group_size
    )
    error = cloned_input - output
    # error = error.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, sep_dim * num_head)
    # num_groups = (sep_dim * num_head) // group_size
    # error = error.view(bsz, seq_len, num_groups, group_size)
    
    if (use_error):
        error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,
 
                                )
        # error_lr = error_lr.view(bsz, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
        
        print("using error")

        return output + error_lr

    return output

def compress_insert_function(
    previous_key,
    previous_value,
    compress_config,
    layer_idx,
    pbase1=None,
    qbase1=None,
    pbase2=None,
    qbase2=None,
    prefill=None,
):
    
    #print("reached compress_insert_function")

    #print("compress_insert_function layer_idx: ", layer_idx)

    batch, num_head, seq_len, sep_dim = previous_key.shape
    if compress_config.token_preserving[layer_idx] == True:
        starting_idx = int(compress_config.start_saving[layer_idx] * seq_len)
        locality_idx = int(compress_config.locality_saving[layer_idx] * seq_len)
    else:
        starting_idx = int(0)
        locality_idx = -seq_len
    # print("starting_idx:", starting_idx, "locality_idx:", locality_idx,compress_config.token_preserving[layer_idx],batch, num_head, seq_len, sep_dim)
    
    if compress_config.compress_method[layer_idx] == "KCVT":
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            seq_len,
        )
        if previous_value is not None:
            previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
                int(num_head * sep_dim),
            )

    if compress_config.compress_method[layer_idx] == "KIVI_V2":
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )
        previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
            previous_value[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )

    if compress_config.compress_method[layer_idx] == "GEAR":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        
        #print("previous_key: ", previous_key.shape)

        previous_key = gearslkivi_channelQ_new(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            compress_config.left[layer_idx],
            rank_used,
            compress_config.loop[layer_idx],
            layer_idx,
            True
        )
        previous_key = previous_key.half()
        #print("previous_val gearslkivi_channelQ_new(): ", previous_value.size())
        previous_value = gearslkivi_tokenQ_new(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            compress_config.left[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx],
            layer_idx,
            False
        )
        #print("previous_value: ", previous_value.size())

        previous_value = previous_value.half()
    if compress_config.compress_method[layer_idx] == "GEAR-KCVT":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = gearslkivi_channelQ_new(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            seq_len,
            compress_config.left[layer_idx],
            rank_used,
            compress_config.loop[layer_idx]
            
        )
        previous_key = previous_key.half()
        previous_value = gearslkivi_tokenQ_new(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            int(num_head * sep_dim),
            compress_config.left[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx]
        )
        previous_value = previous_value.half()
    if compress_config.compress_method[layer_idx] == "GEARL":

        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = tokenwise_gearlkivi_channelQ(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            rank_used,
            compress_config.loop[layer_idx],

            
        )
        previous_value = tokenwise_gearlkivi_tokenQ(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx],
            rankv_used,
            compress_config.loop[layer_idx],
 
        )
    if compress_config.compress_method[layer_idx] == "GEARL-KCVT":
        prefill_rank = int(compress_config.prefill_rank[layer_idx])
        prefill_rankv = int(compress_config.prefill_rankv[layer_idx])
        rank = int(compress_config.rank[layer_idx])
        rankv = int(compress_config.rankv[layer_idx])
        if prefill is True:
            rank_used = prefill_rank
            rankv_used = prefill_rankv
        else:
            rank_used = rank
            rankv_used = rankv
        previous_key = tokenwise_gearlkivi_channelQ(
            previous_key,
            compress_config.quantize_bit[layer_idx],
            seq_len,
            rank_used,
            compress_config.loop[layer_idx],
            
            
        )
        previous_value = tokenwise_gearlkivi_tokenQ(
            previous_value,
            compress_config.quantize_bit[layer_idx],
            int(num_head * sep_dim),
            rankv_used,
            compress_config.loop[layer_idx],
            
        )

    return previous_key, previous_value




