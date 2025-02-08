from icecream import ic

term = 4

##########################################
#  Cloud compute cost calculation
##########################################
rate                = 32.77
transfer_per_month  = 5_000 # [GB]
hours_per_day       = 8
days_per_month      = 22

total_hours             = hours_per_day * days_per_month * 12 * term

base_cost_on_cloud      = total_hours * rate

transfer_cost_per_month = transfer_per_month * 0.09

total_transfer_cost     = transfer_cost_per_month * 12 * term

total_cost_on_cloud     = base_cost_on_cloud + total_transfer_cost

ic(total_cost_on_cloud)

##########################################
# On Premise Cost Calculation
##########################################

cost_per_server      = 65_800
num_servers_required = 2
misc_cost_per_year   = 5_000

total_misc_cost = misc_cost_per_year * num_servers_required * term

total_cost      = cost_per_server * num_servers_required + total_misc_cost

ic(total_cost)
