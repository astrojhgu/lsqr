/// translated from 
/// https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/master/src/lsqr.jl
/// almost line by line

use std::fmt::Debug;
use ndarray::{Array1, ArrayView1, ArrayViewMut1, ScalarOperand};
use num::Float;

fn norm<T: ScalarOperand + Float>(x: ArrayView1<T>) -> T {
    x.map(|&x1| x1 * x1).sum().sqrt()
}

#[allow(clippy::too_many_arguments)]
pub fn lsqr<T, F, FT>(
    mut x: ArrayViewMut1<T>,
    aprod: &F,
    atprod: &FT,
    b: ArrayView1<T>,
    atol: T,
    btol: T,
    conlim: T,
    maxiter: usize,
) where
    T: ScalarOperand + Float + Debug,
    F: Fn(ArrayView1<T>) -> Array1<T>,
    FT: Fn(ArrayView1<T>) -> Array1<T>,
{
    let mut itn = 0;
    let mut istop = 0;
    let ctol = if conlim > T::zero() {
        T::one() / conlim
    } else {
        T::zero()
    };
    let mut acond;
    let mut xnorm;
    let mut anorm = T::zero();
    let mut ddnorm = T::zero();
    let mut res2 = T::zero();
    let mut xxnorm = T::zero();
    let mut z = T::zero();
    let mut sn2 = T::zero();
    let mut cs2 = -T::one();
    let damp = T::zero();
    let dampsq = damp.powi(2);
    let mut u = &b - &aprod(x.view());
    //println!("{:?}", u);
    let mut v = x.to_owned();
    //println!("{:?}", v);
    let beta = norm(u.view());

    let mut alpha = T::zero();
    if beta > T::zero() {
        u = u / beta;
        v = atprod(u.view());
        alpha = norm(v.view());
    }
    //println!("u={:?},alpha={:?}", u, alpha);
    if alpha > T::zero() {
        v = v / alpha;
    }
    //println!("v={:?}", v);
    let mut w = v.clone();
    let mut arnorm = alpha * beta;
    if arnorm == T::zero() {
        return;
    }
    //println!("arnorm={:?}", arnorm);
    let mut rhobar = alpha;
    let mut rnorm;
    let (mut phibar, bnorm) = (beta, beta);
    //println!("phibar={:?}", phibar);
    while itn < maxiter && istop ==0 {
        itn += 1;
        let tmpm = aprod(v.view());
        u = &tmpm - (&u * alpha);
        let beta = norm(u.view());
        //println!("beta={:?}", beta);

        if beta > T::zero() {
            u = u / beta;
            anorm = (anorm.powi(2) + alpha.powi(2) + beta.powi(2) + dampsq).sqrt();
            let tmpn = atprod(u.view());
            v = &tmpn - &v * beta;
            alpha = norm(v.view());
            if alpha > T::zero() {
                v = v / alpha;
            }
        }

        let rhobar1 = (rhobar.powi(2) + dampsq).sqrt();
        
        let cs1 = rhobar / rhobar1;
        let sn1 = damp / rhobar1;
        let psi = sn1 * phibar;
        phibar = cs1 * phibar;
        let rho = (rhobar1.powi(2) + beta.powi(2)).sqrt();
        let cs = rhobar1 / rho;

        let sn = beta / rho;
        let theta = sn * alpha;

        rhobar = -cs * alpha;
        let phi = cs * phibar;
        phibar = sn * phibar;
        let tau = sn * phi;
        let t1 = phi / rho;

        let t2 = -theta / rho;
        //let x1=;
        x.assign(&(&x + &w * t1));
        //println!("x={:?}", x);
        w = &w * t2 + &v;
        //println!("w={:?}", w);
        let wrho = &w / rho;
        ddnorm = ddnorm + norm(wrho.view());

        let delta = sn2 * rho;
        let gambar = -cs2 * rho;
        let rhs = phi - delta * z;
        let zbar = rhs / gambar;
        xnorm = (xxnorm + zbar.powi(2)).sqrt();
        let gamma = (gambar.powi(2) + theta.powi(2)).sqrt();
        cs2 = gambar / gamma;
        sn2 = theta / gamma;
        z = rhs / gamma;
        xxnorm = xxnorm + z.powi(2);

        acond = anorm * T::sqrt(ddnorm);
        let res1 = phibar.powi(2);
        res2 = res2 + psi.powi(2);
        rnorm = T::sqrt(res1 + res2);
        arnorm = alpha * T::abs(tau);

        let r1sq = rnorm.powi(2) - dampsq * xxnorm;
        let mut _r1norm = T::sqrt(T::abs(r1sq));
        if r1sq < T::zero() {
            _r1norm = -_r1norm;
        }
        let _r2norm = rnorm;

        let test1 = rnorm / bnorm;
        //println!("test1={:?}", test1);
        let test2 = arnorm / (anorm * rnorm);
        let test3 = T::one() / acond;
        let t1 = test1 / (T::one() + anorm * xnorm / bnorm);
        let rtol = btol + atol * anorm * xnorm / bnorm;
        //println!("rtol={:?}", rtol);

        if itn >= maxiter {
            istop = 7
        }
        if T::one() + test3 <= T::one() {
            istop = 6
        }
        if T::one() + test2 <= T::one() {
            istop = 5
        }
        if T::one() + t1 <= T::one() {
            istop = 4
        }

        if test3 <= ctol {
            istop = 3
        }
        if test2 <= atol {
            istop = 2
        }
        if test1 <= rtol {
            istop = 1
        }
        
        if rhobar.abs()<T::epsilon(){
            istop=8
        }
        
    }
    //println!("{}", istop);
}
