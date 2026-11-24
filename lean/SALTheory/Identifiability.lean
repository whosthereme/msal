import SALTheory.Library

namespace SALTheory

def dotF (F : Nat) (a b : Fin F → Int) : Int := sumFin F (fun f => a f * b f)

theorem dotF_comm (F : Nat) (a b : Fin F → Int) : dotF F a b = dotF F b a := by
  unfold dotF
  exact sumFin_congr F _ _ (fun f => Int.mul_comm (a f) (b f))

theorem dotF_comb (F n : Nat) (φ : Fin F → Int) (d : Fin n → Fin F → Int) (c : Fin n → Int) :
    dotF F φ (fun f => sumFin n (fun i => c i * d i f)) = sumFin n (fun i => c i * dotF F φ (d i)) := by
  unfold dotF
  have step1 : (fun f => φ f * sumFin n (fun i => c i * d i f)) =
      (fun f => sumFin n (fun i => c i * (φ f * d i f))) := by
    funext f
    rw [← sumFin_mul_left]
    exact sumFin_congr n _ _ (fun i => Int.mul_left_comm (φ f) (c i) (d i f))
  have step2 : (fun i => c i * sumFin F (fun f => φ f * d i f)) =
      (fun i => sumFin F (fun f => c i * (φ f * d i f))) := by
    funext i
    rw [sumFin_mul_left]
  rw [step1, step2]
  exact sumFin_comm F n (fun f i => c i * (φ f * d i f))

theorem sumFin_nonneg (n : Nat) (f : Fin n → Int) (h : ∀ i, 0 ≤ f i) : 0 ≤ sumFin n f := by
  induction n with
  | zero => exact Int.le_refl 0
  | succ n ih =>
      have h1 := ih (fun i => f i.castSucc) (fun i => h i.castSucc)
      have h2 := h (Fin.last n)
      simp only [sumFin]
      omega

theorem shift_invisible (F : Nat) (φ q d : Fin F → Int) (t : Int) (hq : dotF F q d = 0) :
    dotF F (fun f => φ f + t * q f) d = dotF F φ d := by
  unfold dotF at *
  have pw : (fun f => (φ f + t * q f) * d f) = (fun f => φ f * d f + t * (q f * d f)) := by
    funext f
    rw [Int.add_mul, Int.mul_assoc]
  rw [pw, sumFin_add, sumFin_mul_left, hq, Int.mul_zero, Int.add_zero]

theorem shift_moves_held (F : Nat) (φ q dstar : Fin F → Int) (t : Int) :
    dotF F (fun f => φ f + t * q f) dstar = dotF F φ dstar + t * dotF F q dstar := by
  unfold dotF
  have pw : (fun f => (φ f + t * q f) * dstar f) = (fun f => φ f * dstar f + t * (q f * dstar f)) := by
    funext f
    rw [Int.add_mul, Int.mul_assoc]
  rw [pw, sumFin_add, sumFin_mul_left]

theorem held_conditioning_not_identified {X : Type} (F m n : Nat) (Φ : Fin m → Fin F → Int) (q : Fin F → Int)
    (t : Fin m → Int) (train : Fin n → Fin F → Int) (dstar : Fin F → Int) (G : X → (Fin m → Int) → Int)
    (horth : ∀ i, dotF F q (train i) = 0) :
    (∀ x i, G x (fun j => dotF F (fun f => Φ j f + t j * q f) (train i)) = G x (fun j => dotF F (Φ j) (train i))) ∧
    (∀ j, dotF F (fun f => Φ j f + t j * q f) dstar = dotF F (Φ j) dstar + t j * dotF F q dstar) := by
  constructor
  · intro x i
    have h : (fun j => dotF F (fun f => Φ j f + t j * q f) (train i)) = (fun j => dotF F (Φ j) (train i)) := by
      funext j
      exact shift_invisible F (Φ j) q (train i) (t j) (horth i)
    rw [h]
  · intro j
    exact shift_moves_held F (Φ j) q dstar (t j)

theorem span_determined (F n : Nat) (φ ψ : Fin F → Int) (train : Fin n → Fin F → Int) (c : Fin n → Int)
    (hagree : ∀ i, dotF F φ (train i) = dotF F ψ (train i)) :
    dotF F φ (fun f => sumFin n (fun i => c i * train i f)) =
      dotF F ψ (fun f => sumFin n (fun i => c i * train i f)) := by
  rw [dotF_comb, dotF_comb]
  exact sumFin_congr n _ _ (fun i => by rw [hagree i])

theorem gradient_orthogonal (F n : Nat) (train : Fin n → Fin F → Int) (g : Fin n → Int) (q : Fin F → Int)
    (horth : ∀ i, dotF F q (train i) = 0) :
    dotF F (fun f => sumFin n (fun i => g i * train i f)) q = 0 := by
  rw [dotF_comm, dotF_comb]
  have h : (fun i => g i * dotF F q (train i)) = (fun _ => 0) := by
    funext i
    rw [horth i, Int.mul_zero]
  rw [h, sumFin_zero]

theorem orthogonal_shift_norm (F : Nat) (p q : Fin F → Int) (s : Int) (hpq : dotF F p q = 0) :
    dotF F (fun f => p f + s * q f) (fun f => p f + s * q f) = dotF F p p + s * s * dotF F q q := by
  unfold dotF at *
  have pw : (fun f => (p f + s * q f) * (p f + s * q f)) =
      (fun f => (p f * p f + s * (p f * q f)) + (s * (p f * q f) + s * s * (q f * q f))) := by
    funext f
    simp only [Int.add_mul, Int.mul_add]
    ac_rfl
  rw [pw, sumFin_add, sumFin_add, sumFin_add, sumFin_mul_left, sumFin_mul_left, hpq]
  simp only [Int.mul_zero, Int.add_zero, Int.zero_add]

theorem min_norm_at_zero_shift (F : Nat) (p q : Fin F → Int) (s : Int) (hpq : dotF F p q = 0) :
    dotF F p p ≤ dotF F (fun f => p f + s * q f) (fun f => p f + s * q f) := by
  rw [orthogonal_shift_norm F p q s hpq]
  have hq : 0 ≤ dotF F q q := sumFin_nonneg F _ (fun f => mul_self_nonneg' (q f))
  have hs : 0 ≤ s * s * dotF F q q := Int.mul_nonneg (mul_self_nonneg' s) hq
  omega

def oneHot (F : Nat) (k : Fin F) : Fin F → Int := fun f => if f = k then 1 else 0

theorem one_hot_orthogonal (F : Nat) (i k : Fin F) (hik : i ≠ k) : dotF F (oneHot F i) (oneHot F k) = 0 := by
  unfold dotF oneHot
  have h : (fun f => (if f = i then (1 : Int) else 0) * (if f = k then 1 else 0)) = (fun _ => 0) := by
    funext f
    by_cases hi : f = i
    · simp [hi, hik]
    · simp [hi]
  rw [h, sumFin_zero]

theorem sumFin_single (n : Nat) (f : Fin n → Int) (k : Fin n) (h : ∀ i, i ≠ k → f i = 0) : sumFin n f = f k := by
  induction n with
  | zero => exact k.elim0
  | succ n ih =>
      simp only [sumFin]
      by_cases hl : k = Fin.last n
      · have hz : (fun i : Fin n => f i.castSucc) = (fun _ => 0) := by
          funext i
          apply h
          intro he
          rw [hl] at he
          have hv := congrArg Fin.val he
          have := i.isLt
          simp at hv
          omega
        rw [hz, sumFin_zero, hl, Int.zero_add]
      · have hlt : k.val < n := by
          have := k.isLt
          have hne : k.val ≠ n := fun he => hl (Fin.ext he)
          omega
        have hlast : f (Fin.last n) = 0 := h _ (fun he => hl he.symm)
        have ih' := ih (fun i => f i.castSucc) ⟨k.val, hlt⟩ (fun i hi => h _ (fun he => hi (Fin.ext (by
          have hv := congrArg Fin.val he
          simpa using hv))))
        rw [ih', hlast, Int.add_zero]
        all_goals exact congrArg f (Fin.ext rfl)

theorem one_hot_self (F : Nat) (k : Fin F) : dotF F (oneHot F k) (oneHot F k) = 1 := by
  unfold dotF
  rw [sumFin_single F _ k (fun i hi => by simp [oneHot, hi])]
  simp [oneHot]

theorem flow_error_is_estimate_error (N p ε a u : Int) :
    p * (u - (ε - a)) = N * a - ((p * ε + (N - p) * a) - p * u) := by
  rw [Int.mul_sub, Int.mul_sub, Int.sub_mul]
  omega

theorem euler_step_mixes (x u τ Δ : Int) : τ * (x - Δ * u) = (τ - Δ) * x + Δ * (x - τ * u) := by
  rw [Int.mul_sub, Int.sub_mul, Int.mul_sub, Int.mul_left_comm Δ τ u]
  omega

end SALTheory
