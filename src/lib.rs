extern crate rand;

use rand::prelude::*;
use rand::{thread_rng, Rng};
use rand::distributions::Normal;

struct upper_lower {
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
    ul:&Vec<upper_lower>,
    uniform_rand_generator:impl Fn()->f64
)->Vec<f64>{
    ul.iter().map(|v|get_random_parameter(v.lower, v.upper, uniform_rand_generator())).collect()
}

fn get_new_paramater_and_fn(
    ul:&Vec<upper_lower>,
    obj_fn:&impl Fn(&Vec<f64>)->f64,
    uniform_rand_generator:&impl Fn()->f64
)->(Vec<f64>, f64) {
    let parameters=get_random_parameters(ul, uniform_rand_generator);
    let fn_value_at_parameters=obj_fn(&parameters);
    (parameters, fn_value_at_parameters)
}

static step_increment:f64=0.01;
fn get_step_size(curr:f64, best:f64, lower:f64, upper:f64)->f64{
    step_increment*(upper-lower)*(curr-best)
}

fn get_new_nests(
    ul:&Vec<upper_lower>, 
    obj_fn:&impl Fn(&Vec<f64>)->f64,
    uniform_rand_generator:&impl Fn()->f64,
    n:usize
)->Vec<(Vec<f64>, f64)>{
    (0..n).map(|_|get_new_paramater_and_fn(ul, &obj_fn, &uniform_rand_generator)).collect()
}

fn get_cuckoos(
    new_nest:&Vec<(Vec<f64>, f64)>,
    curr_nest:&Vec<(Vec<f64>, f64)>, //see if can move for efficiency (return self)
    best_parameters:&Vec<f64>,
    obj_fn:impl Fn(&Vec<f64>)->f64,
    ul:&Vec<upper_lower>,
    lambda:f64,
    uniform_rand_generator:impl Fn()->f64,
    normal_rand_generator:impl Fn()->f64
)->Vec<(Vec<f64>, f64)>
{
    curr_nest.iter().map(|(parameters, _)|{
        let new_nest_parameters=parameters.iter().zip(ul.iter()).zip(best_parameters.iter()).map(|((param, v), bp)|{
            get_truncated_parameter(
                v.lower, v.upper, 
                get_levy_flight(
                    *param, 
                    get_step_size(*param, *bp, v.lower, v.upper), 
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
    ul:&Vec<upper_lower>,
    p:f64
)->Vec<(Vec<f64>, f64)>{
    let n=new_nest.len();
    let num_to_keep=((n as f64)*p) as usize;
    let start_num=n-num_to_keep;
    new_nest.into_iter().enumerate().map(|(index, v)|{
        if index<start_num {v} else {get_new_paramater_and_fn(ul, &obj_fn, &uniform_rand_generator)}
    }).collect()
}

pub fn optimize(
    obj_fn:&impl Fn(&Vec<f64>)->f64,
    ul:&Vec<upper_lower>,
    n:usize,
    total_mc:usize,
    tol:f64,
    seed:usize
)->(Vec<f64>, f64){
    let normal = Normal::new(0.0, 1.0);
    let normal_rand_generator=||normal.sample(&mut rand::thread_rng());
    let uniform_rand_generator=||rng.gen();
    //let v = normal.sample(&mut rand::thread_rng());
}

/** template< typename Array, typename ObjFn>
    auto optimize(const ObjFn& objFn, const Array& ul, int n, int totalMC, double tol, int seed){
        int numParams=ul.size();
        srand(seed);
        SimulateNorm norm(seed);
        auto unifL=[](){return swarm_utils::getUniform();};
        auto normL=[&](){return norm.getNorm();};
        auto nest=getNewNest(ul, objFn,normL, n);
        double lambda=1.5;
        double pMin=.05;
        double pMax=.5;
        
        double fMin=2;
        int i=0;

        sortNest(nest);
        auto newNest=getNewNest(ul, objFn,normL, n);
       
       
        while(i<totalMC&&fMin>tol){
            /**Completely overwrites newNest*/
            //newNest now has the previous values from nest with levy flights added
            getCuckoos(
                &newNest, 
                nest, nest[0].first, //the current best nest
                objFn, ul, 
                lambda, 
                unifL, 
                normL
            );
            //compare previous nests with cuckoo nests and sort results
            //nest now has the best of nest and newNest
            getBestNest(
                &nest, 
                newNest
            );
            //remove bottom "p" nests and resimulate.
            emptyNests(&nest, objFn, normL, ul, getPA(pMin, pMax, i, totalMC));
            sortNest(nest);
            fMin=nest[0].second;

            #ifdef VERBOSE_FLAG
                std::cout<<"Index: "<<i<<", Param Vals: ";
                for(auto& v:nest[0].first){
                    std::cout<<v<<", ";
                }
                std::cout<<", Obj Val: "<<fMin<<std::endl;
            #endif
            ++i;
        }
        return nest[0];
} */


//    let normal = Normal::new(0.0, 1.0);
//let v = normal.sample(&mut rand::thread_rng());


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
