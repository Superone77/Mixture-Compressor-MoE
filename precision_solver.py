import pickle
import os
import argparse
import csv

import torch  
import gurobipy as gp

class experts_ilp():
    def __init__(self, 
                 actnum_path, 
                 x_space=(1,2,3), 
                 num_experts=8, 
                 quant_loss_path=None, 
                 weight_path=None, 
                 alpha=1,
                 beta=1,
                 gama=1,
                 norm_experts=False
                 ):
        self.x_space = x_space
        self.num_experts = num_experts
        
        with open(actnum_path, 'rb') as file:
            actnum_matrix = pickle.load(file)
        with open(quant_loss_path, 'rb') as file:
            quant_loss_matrix = pickle.load(file)
        with open(weight_path, 'rb') as file:
            weight_matrix = pickle.load(file)
        
        self.blocks = list(actnum_matrix.keys())
        self.actnum_matrix_raw = actnum_matrix
        scale_factor = 1
        if norm_experts:
            actnum_matrix = self.norm_experts_dim(actnum_matrix)
            weight_matrix = self.norm_experts_dim(weight_matrix)
            scale_factor = 1000
        self.loss_matrix = {}
        for i in self.blocks:
            i_loss_matrix = {}
            for j in range(self.num_experts):
                j_loss_matrix = {}
                expert_significance = actnum_matrix[i][j] ** alpha * weight_matrix[i][j] ** beta
                for x in self.x_space:
                    j_loss_matrix[x] = expert_significance  * quant_loss_matrix[i][j][x] ** alpha * scale_factor
                i_loss_matrix[j] = j_loss_matrix
            self.loss_matrix[i] = i_loss_matrix                     
        
    def bulid_ilp_model(self, nblock, constrait):
        loss_matrix = self.loss_matrix[nblock]
        lp_content = "Minimize\nOBJ"
        lp_content += "\nSubject To\n"
        lp_content += " + ".join(f"y{i}" for i in range(1, self.num_experts + 1)) + " - OBJ = 0\n"
        lp_content += " + ".join(f"1 x{i}_{1} + 2 x{i}_{2} + 3 x{i}_{3}" for i in range(1, self.num_experts + 1)) + f" <= {constrait}\n"
        lp_content += " + ".join(f"x{i}_{3}" for i in range(1, self.num_experts + 1)) + f" >= 1\n"
        lp_content += " + ".join(f"x{i}_{2}" for i in range(1, self.num_experts + 1)) + f" >= 1\n"
        for i in range(1, self.num_experts + 1):
            lp_content += f"y{i} - " + " - ".join(f"{loss_matrix[i-1][j]} x{i}_{j}" for j in self.x_space) + " = 0\n"
            lp_content += f" + ".join(f"x{i}_{j}" for j in self.x_space) + " = 1\n"
        lp_content += "Binary\n"
        lp_content += " ".join(f"x{i}_{j}" for i in range(1, self.num_experts + 1) for j in self.x_space)
        return lp_content

    def solve_ilp_model(self, model_path):
        model = gp.read(model_path)
        model.optimize()
        opt_set = []
        for v in model.getVars():
            if v.VarName.startswith('x'):
                if v.X == 1:
                    opt_set.append(int(v.VarName[-1]))
        experts_keys = list(range(self.num_experts))
        opt_set_dict = dict(zip(experts_keys, opt_set))
        return opt_set_dict
    
    def expert2tensor(self, expert_dict):
        experts_tensor = torch.tensor(list(expert_dict.values()))
        return experts_tensor
    
    def norm_experts_dim(self, x):
        norm_x = {}
        for i in self.blocks:
            if not torch.is_tensor(x[i]):
                experts_tensor = self.expert2tensor(x)
            else:
                experts_tensor = x[i]
            norm_experts = experts_tensor / float(experts_tensor.sum())
            norm_x[i] = norm_experts
        return norm_x
                  
    def ilp_solver(self, constrait):
        final_opt_set = {}
        for n in self.blocks:
            lp_model = self.bulid_ilp_model(n, constrait)
            with open('model.lp', 'w') as file:
                file.write(lp_model)
            opt_set = self.solve_ilp_model('model.lp')
            final_opt_set[n] = opt_set
        return final_opt_set

    def save_experts_csv(self, opt_set, csv_path, actnum_matrix=None):
        if actnum_matrix is None:
            actnum_matrix = self.actnum_matrix_raw
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["block", "expert", "activation_frequency", "assigned_precision"])
            for block in sorted(actnum_matrix.keys()):
                freqs = actnum_matrix[block]
                for expert_id in range(self.num_experts):
                    freq_val = freqs[expert_id]
                    if torch.is_tensor(freq_val):
                        freq_val = freq_val.item()
                    else:
                        freq_val = float(freq_val)
                    writer.writerow([block, expert_id, freq_val, opt_set[block][expert_id]])

def get_args_parser():
    parser = argparse.ArgumentParser('Set ilp configs', add_help=False)
    parser.add_argument('--actnum_path', default='experts_act_frequency.pkl', type=str)
    parser.add_argument('--quant_loss_path', default='experts_quant_loss.pkl', type=str)
    parser.add_argument('--weight_path',default='experts_act_weight.pkl', type=str)
    parser.add_argument('--save_path',default='experts_mixture_bit_selection', type=str)
    parser.add_argument('--csv_save_path', default=None, type=str)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--beta', default=1.5, type=float)
    parser.add_argument('--gama', default=2, type=float)
    parser.add_argument('--start_bitwidth', default=12, type=int)
    parser.add_argument('--end_bitwidth', default=21, type=int)

    return parser
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Experts bit selection with ilp', parents=[get_args_parser()])
    args = parser.parse_args()
    experts_ilp_example = experts_ilp(args.actnum_path, 
                                      quant_loss_path=args.quant_loss_path,
                                      weight_path=args.weight_path,
                                      alpha=args.alpha,
                                      beta=args.beta,
                                      gama=args.gama,
                                      norm_experts=True)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    csv_save_path = args.csv_save_path or args.save_path
    if not os.path.exists(csv_save_path):
        os.makedirs(csv_save_path)
    for i in range(args.start_bitwidth, args.end_bitwidth):
        opt_set = experts_ilp_example.ilp_solver(i)
        # total bits of each MoE block, average bits can be calculated as total_bits / 8
        total_bits = str(i)
        save_name = f"experts_mixture_bitwidth_combination_{total_bits}bit.pkl"
        with open(os.path.join(args.save_path, save_name), 'wb') as f:
            pickle.dump(opt_set, f) 
        csv_name = f"experts_mixture_bitwidth_combination_{total_bits}bit.csv"
        experts_ilp_example.save_experts_csv(opt_set, os.path.join(csv_save_path, csv_name))
