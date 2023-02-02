use ndarray::{Array1, ArrayView1, ArrayViewMut1, ScalarOperand};
use num::Float;
/// translated from
/// https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/master/src/lsqr.jl
/// almost line by line
use std::fmt::Debug;

fn norm<T: ScalarOperand + Float>(x: ArrayView1<T>) -> T {
    x.map(|&x1| x1 * x1).sum().sqrt()
}

fn abs2<T: ScalarOperand + Float>(x: T) -> T {
    x.powi(2)
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
    let mut rho_bar = alpha;
    let mut rnorm;
    let (mut phibar, bnorm) = (beta, beta);
    //println!("phibar={:?}", phibar);
    while itn < maxiter && istop == 0 {
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

        let rho_bar1 = (rho_bar.powi(2) + dampsq).sqrt();

        let cs1 = rho_bar / rho_bar1;
        let sn1 = damp / rho_bar1;
        let psi = sn1 * phibar;
        phibar = cs1 * phibar;
        let rho = (rho_bar1.powi(2) + beta.powi(2)).sqrt();
        let cs = rho_bar1 / rho;

        let sn = beta / rho;
        let theta = sn * alpha;

        rho_bar = -cs * alpha;
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
        let z_bar = rhs / gambar;
        xnorm = (xxnorm + z_bar.powi(2)).sqrt();
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

        if rho_bar.abs() < T::epsilon() {
            istop = 8
        }
    }
    //println!("{}", istop);
}

#[allow(clippy::too_many_arguments)]
pub fn lsmr<T, F, FT>(
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
    let lambda=T::zero();
    let ctol=T::one()/conlim;
    let tmp_u=aprod(x.view());
    let b=&b-&tmp_u;
    let mut u=b.clone();
    let mut beta=norm(u.view());
    u=u/beta;
    let mut v=atprod(u.view());
    let mut alpha=norm(v.view());
    v=v/alpha;
    let mut z_bar=alpha*beta;
    let mut alpha_bar=alpha;
    let mut rho=T::one();
    let mut rho_bar=T::one();
    let mut c_bar=T::one();
    let mut s_bar=T::zero();
    let mut h=v.clone();
    let mut h_bar=Array1::<T>::zeros(x.len());
    let mut beta_dd=beta;
    let mut beta_d=T::zero();
    let mut rho_d_old=T::one();
    let mut tau_tilde_old=T::zero();
    let mut theta_tilde=T::zero();
    let mut z=T::zero();
    let mut d=T::zero();

    let mut norma;
    let mut conda;
    let mut normx;

    let mut norma2=abs2(alpha);
    let mut max_rho_bar=T::zero();
    let mut min_rho_bar=T::from(1e100).unwrap();

    let normb=beta;
    let mut istop=0;
    let mut normr;
    let mut normar=alpha*beta;
    let mut iter=0;
    if normar!=T::zero(){
        while iter<maxiter{
            iter+=1;
            let tmp_u=aprod(v.view());
            u=tmp_u-&u*alpha;
            beta=norm(u.view());

            if beta>T::zero(){
                u=&u/beta;
                let tmp_v=atprod(u.view());
                v=&tmp_v-v*beta;
                alpha=norm(v.view());
                v=v/alpha;
            }

            let alpha_hat=alpha_bar.hypot(lambda);
            let c_hat=alpha_bar/alpha_hat;
            let s_hat=lambda/alpha_hat;

            let rho_old=rho;
            rho=alpha_hat.hypot(beta);
            let c=alpha_hat/rho;
            let s=beta/rho;
            let theta_new=s*alpha;
            alpha_bar=c*alpha;


            let rho_bar_old=rho_bar;
            let zold=z;
            let theta_bar=s_bar*rho;
            let rho_temp=c_bar*rho;
            rho_bar=(c_bar*rho).hypot(theta_new);
            c_bar=c_bar*rho/rho_bar;
            s_bar = theta_new / rho_bar;
            z = c_bar * z_bar;
            z_bar = - s_bar * z_bar;

            h_bar = h_bar * (-theta_bar * rho / (rho_old * rho_bar_old)) + &h;
            x.assign(&(&x+  &h_bar*(z / (rho * rho_bar))));
            h = &h * (-theta_new / rho) + &v;


            let beta_acute = c_hat * beta_dd;
            let beta_check = - s_hat * beta_dd;

            
            let beta_hat = c * beta_acute;
            beta_dd = - s * beta_acute;

            
            let theta_tilde_old = theta_tilde;
            let rho_tilde_old = T::hypot(rho_d_old, theta_bar);
            let c_tilde_old = rho_d_old / rho_tilde_old;
            let s_tilde_old = theta_bar / rho_tilde_old;
            theta_tilde = s_tilde_old * rho_bar;
            rho_d_old = c_tilde_old * rho_bar;
            beta_d = - s_tilde_old * beta_d + c_tilde_old * beta_hat;

            tau_tilde_old = (zold - theta_tilde_old * tau_tilde_old) / rho_tilde_old;
            let tau_d = (z - theta_tilde * tau_tilde_old) / rho_d_old;
            d =d+ abs2(beta_check);
            normr = T::sqrt(d + abs2(beta_d - tau_d) + abs2(beta_dd));

            norma2 =norma2+ abs2(beta);
            norma  = T::sqrt(norma2);
            norma2 = norma2+abs2(alpha);

            
            max_rho_bar = T::max(max_rho_bar, rho_bar_old);
            if iter > 1{
                min_rho_bar = T::min(min_rho_bar, rho_bar_old);
            }
                
            conda = T::max(max_rho_bar, rho_temp) / T::min(min_rho_bar, rho_temp);

            normar  = T::abs(z_bar);
            normx = norm(x.view());

            let test1 = normr / normb;
            let test2 = normar / (norma * normr);
            let test3 = T::one()/conda;
            

            let t1 = test1 / (T::one() + norma * normx / normb);
            let rtol = btol + atol * norma * normx / normb;
            
            if iter >= maxiter {istop = 7; break}
            if T::one() + test3 <= T::one() {istop = 6; break}
            if T::one() + test2 <= T::one() {istop = 5; break}
            if T::one() + t1 <= T::one() {istop = 4; break}
            
            if test3 <= ctol {istop = 3; break}
            if test2 <= atol {istop = 2; break}
            if test1 <= rtol  {istop = 1; break}
        }
    }
    println!("istop={}", istop);

}
