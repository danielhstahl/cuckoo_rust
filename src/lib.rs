extern crate rand;

use rand::prelude::*;
use rand::{thread_rng, SeedableRng, Rng, StdRng};
use rand::distributions::Normal;

struct Upper_lower {
    lower: f64,
    upper: f64,
}

fn get_levy(alpha:f64, uniform_rand:f64)->f64 {
    uniform_rand.powf(-1.0/alpha)
}

fn get_levy_flight(
    curr_val:f64,
    step_size:f64,
    lambda:f64,
    uniform_rand:f64,
    norm_rand:f64
)->f64 {
    curr_val+step_size*get_levy(lambda, uniform_rand)*norm_rand
}

fn get_random_parameter(
    lower:f64,
    upper:f64,
    uniform_rand:f64
)->f64{
    let half=0.5;
    (upper+lower)*half+(upper-lower)*half*uniform_rand //reflects middle more likely than edges
}

fn get_truncated_parameter(
    lower:f64,
    upper:f64,
    result:f64
)->f64 {
    if result>upper {upper} else if result<lower {lower} else {result}
}

fn get_random_parameters(
    ul:&Vec<Upper_lower>,
    rand_generator:impl Fn()->f64
)->Vec<f64>{
    ul.iter().map(|v|get_random_parameter(v.lower, v.upper, rand_generator())).collect()
}

fn get_new_paramater_and_fn(
    ul:&Vec<Upper_lower>,
    obj_fn:&impl Fn(&Vec<f64>)->f64,
    rand_generator:&impl Fn()->f64
)->(Vec<f64>, f64) {
    let parameters=get_random_parameters(ul, rand_generator);
    let fn_value_at_parameters=obj_fn(&parameters);
    (parameters, fn_value_at_parameters)
}

static step_increment:f64=0.01;
fn get_step_size(curr:f64, best:f64, lower:f64, upper:f64)->f64{
    step_increment*(upper-lower)*(curr-best)
}

fn get_new_nest(
    ul:&Vec<Upper_lower>, 
    obj_fn:&impl Fn(&Vec<f64>)->f64,
    rand_generator:&impl Fn()->f64,
    n:usize
)->Vec<(Vec<f64>, f64)>{
    (0..n).map(|_|get_new_paramater_and_fn(ul, &obj_fn, &rand_generator)).collect()
}

fn sort_nest(
    nest:Vec<(Vec<f64>, f64)>//move nest
)->Vec<(Vec<f64>, f64)>{
    let mut tmp_nest=nest.sort_by(|(_, a), (_, b)| a<b); //smallest to largest...I hope the compiler optimizes this
    tmp_nest
}

fn get_best_nest(
    new_nest:&Vec<(Vec<f64>, f64)>,
    curr_nest:Vec<(Vec<f64>, f64)>//move curr_nest
)->Vec<(Vec<f64>, f64)> {
    sort_nest(curr_nest.into_iter().zip(new_nest.iter()).map(|(curr_val, new_val)|{
        let (curr_params, curr_fn_val)=curr_val;
        let (new_params, new_fn_val)=new_val;
        if new_fn_val< curr_fn_val { new_val } else { curr_val }
    }).collect())
}

fn get_cuckoos(
    new_nest:Vec<(Vec<f64>, f64)>, //move for efficiency (return self)
    curr_nest:&Vec<(Vec<f64>, f64)>,
    best_parameters:&Vec<f64>,
    ul:&Vec<Upper_lower>,
    obj_fn:impl Fn(&Vec<f64>)->f64,
    lambda:f64,
    uniform_rand_generator:impl Fn()->f64,
    normal_rand_generator:impl Fn()->f64
)->Vec<(Vec<f64>, f64)>
{
    new_nest.into_iter()
        .zip(curr_nest.iter())
        .map(|((_, _), (curr_parameters, _))|{
            let new_nest_parameters=curr_parameters.iter()
                .zip(ul.iter())
                .zip(best_parameters.iter())
                .map(|((curr_param, v), bp)|{
                    get_truncated_parameter(
                        v.lower, v.upper, 
                        get_levy_flight(
                            *curr_param, 
                            get_step_size(*curr_param, *bp, v.lower, v.upper), 
                            lambda, 
                            uniform_rand_generator(), 
                            normal_rand_generator()
                        )
                    )
                }).collect();
            
            let new_nest_fn=obj_fn(&new_nest_parameters);
            (new_nest_parameters, new_nest_fn)
        }).collect()
}

fn get_pa(
    p_min:f64,
    p_max:f64,
    index:usize,
    n:usize
)->f64{
    p_max-(p_max-p_min)*(index as f64)/(n as f64)
}

fn empty_nests(
    new_nest:Vec<(Vec<f64>, f64)>, //move this for efficiency (can return self)
    obj_fn:&impl Fn(&Vec<f64>)->f64,
    uniform_rand_generator:&impl Fn()->f64,
    ul:&Vec<Upper_lower>,
    p:f64
)->Vec<(Vec<f64>, f64)>{
    let n=new_nest.len();
    let num_to_keep=((n as f64)*p) as usize;
    let start_num=n-num_to_keep;
    new_nest.into_iter().enumerate().map(|(index, v)|{
        if index<start_num {v} else {get_new_paramater_and_fn(ul, &obj_fn, &uniform_rand_generator)}
    }).collect()
}

fn get_rng_seed(seed:&[usize])->StdRng{
    let mut rng: StdRng = SeedableRng::from_seed(seed); 
    rng
}

fn get_rng_system_seed()->StdRng{
    let mut rng: StdRng=thread_rng();
    rng
}

pub fn optimize(
    obj_fn:&impl Fn(&Vec<f64>)->f64,
    ul:&Vec<Upper_lower>,
    n:usize,
    total_mc:usize,
    tol:f64,
    rng_inst:impl Fn()->StdRng,
)->(Vec<f64>, f64){
    let mut rng=rng_inst();
    let mu=0.0;
    let sigma=1.0;
    let lambda=1.5;//controls size of levy moves
    let p_min=0.05;//min percentage of nests to replace
    let p_max=0.5;//max percentage of nests to replace

    let normal = Normal::new(mu, sigma);
    let normal_rand_generator=||normal.sample(&mut rng);
    let uniform_rand_generator=||rng.gen();

    let mut curr_nest=sort_nest(get_new_nest(&ul, &obj_fn, &normal_rand_generator, n));

    let mut new_nest=get_new_nest(&ul, &obj_fn, &normal_rand_generator, n);

    let mut done = false; // mut done: bool
    let mut index=0;
    while !done {
        let (curr_best_params, _)=curr_nest.front().unwrap();
        new_nest=get_cuckoos(
            new_nest, &curr_nest, 
            &curr_best_params, &ul, 
            &obj_fn, lambda, 
            &uniform_rand_generator, &normal_rand_generator
        );
        curr_nest=sort_nest(
            empty_nests(
                get_best_nest(&new_nest, curr_nest), 
                &ul, &obj_fn, &normal_rand_generator, 
                get_pa(p_min, p_max, index, total_mc)
            )
        );
        let (_, fn_min)=curr_nest.front().unwrap();
        index=index+1;
        done=index>=total_mc || f_min<=tol; 
    }
    curr_nest.front().unwrap()
}


#[cfg(test)]
mod tests {
    #[test]
    fn simple_fn_optim() {
        let seed: &[_] = &[1, 2, 3, 4];
        let bounds:Upper_lower=Upper_lower{ lower:-4.0, upper:4.0};
        let ul=vec![4, bounds];

        assert_eq!(2 + 2, 4);
    }
}

/**
TEST_CASE("Test Simple Function", "[Cuckoo]"){
    std::vector<swarm_utils::Upper_lower<double> > ul;
    swarm_utils::Upper_lower<double> bounds={-4.0, 4.0};
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    auto results=cuckoo::optimize([](const std::vector<double>& inputs){
        return inputs[0]*inputs[0]+inputs[1]*inputs[1]+inputs[2]*inputs[2]+inputs[3]*inputs[3];
    }, ul, 25, 1000, .00000001, 42);
    auto params=std::get<swarm_utils::optparms>(results);
    std::cout<<params[0]<<", "<<params[1]<<std::endl;
    REQUIRE(std::get<swarm_utils::fnval>(results)==Approx(0.0));
}  
TEST_CASE("Test Rosenbrok Function", "[Cuckoo]"){
    std::vector<swarm_utils::Upper_lower<double> > ul;
    swarm_utils::Upper_lower<double> bounds={-4.0, 4.0};
    ul.push_back(bounds);
    ul.push_back(bounds);

    auto results=cuckoo::optimize([](const std::vector<double>& inputs){
        return futilities::const_power(1-inputs[0], 2)+100*futilities::const_power(inputs[1]-futilities::const_power(inputs[0], 2), 2);
    }, ul, 20, 10000, .00000001, 42);
    auto params=std::get<swarm_utils::optparms>(results);
    //std::cout<<params[0]<<", "<<params[1]<<std::endl;
    REQUIRE(std::get<swarm_utils::fnval>(results)==Approx(0.0));
    //REQUIRE(params[0]==Approx(1.0));
    //REQUIRE(params[1]==Approx(1.0));
}  
TEST_CASE("Test u^2 Function", "[Cuckoo]"){
    std::vector<swarm_utils::Upper_lower<double> > ul;
    swarm_utils::Upper_lower<double> bounds={-5.0, 5.0};
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    int numNests=25;
    int maxMC=25000;
    auto results=cuckoo::optimize([](const std::vector<double>& inputs){
        return futilities::sum(inputs, [](const auto& v, const auto& index){
            return futilities::const_power(v-1.0, 2);
        });
    }, ul, numNests, maxMC, .00000001, 42);
    auto params=std::get<swarm_utils::optparms>(results);
    //std::cout<<params[0]<<", "<<params[1]<<std::endl;
    std::cout<<"obj fn: "<<std::get<swarm_utils::fnval>(results)<<std::endl;
    REQUIRE(std::get<swarm_utils::fnval>(results)==Approx(0.0));
    //REQUIRE(params[0]==Approx(1.0));
    //REQUIRE(params[1]==Approx(1.0));
}  
constexpr double rastigrinScale=10;
TEST_CASE("Test Rastigrin Function", "[Cuckoo]"){
    std::vector<swarm_utils::Upper_lower<double> > ul;
    swarm_utils::Upper_lower<double> bounds={-4.0, 4.0};
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    ul.push_back(bounds);
    int n=ul.size();
    auto results=cuckoo::optimize([](const std::vector<double>& inputs){
        return rastigrinScale*inputs.size()+futilities::sum(inputs, [](const auto& val, const auto& index){
            return futilities::const_power(val, 2)-rastigrinScale*cos(2*M_PI*val);
        });
    }, ul, 25, 10000, .00000001, 42);
    auto params=std::get<swarm_utils::optparms>(results);
    for(auto& v:params){
        std::cout<<v<<",";
    }
    REQUIRE(std::get<swarm_utils::fnval>(results)==Approx(0.0));
    //REQUIRE(params[0]==Approx(1.0));
    //REQUIRE(params[1]==Approx(1.0));
}  */
