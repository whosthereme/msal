import Std

namespace SALTheory

def sumFin : (n : Nat) → (Fin n → Int) → Int
  | 0, _ => 0
  | n + 1, f => sumFin n (fun i => f i.castSucc) + f (Fin.last n)

theorem sumFin_zero (n : Nat) : sumFin n (fun _ => 0) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [sumFin, ih]

theorem sumFin_add (n : Nat) (f g : Fin n → Int) :
    sumFin n (fun i => f i + g i) = sumFin n f + sumFin n g := by
  induction n with
  | zero => simp [sumFin]
  | succ n ih =>
      have h := ih (fun i => f i.castSucc) (fun i => g i.castSucc)
      simp only [sumFin] at h ⊢
      omega

theorem sumFin_mul_left (n : Nat) (c : Int) (f : Fin n → Int) :
    sumFin n (fun i => c * f i) = c * sumFin n f := by
  induction n with
  | zero => simp [sumFin]
  | succ n ih =>
      have h := ih (fun i => f i.castSucc)
      simp only [sumFin] at h ⊢
      rw [h, Int.mul_add]

theorem sumFin_mul_right (n : Nat) (c : Int) (f : Fin n → Int) :
    sumFin n (fun i => f i * c) = sumFin n f * c := by
  have h := sumFin_mul_left n c f
  simp only [Int.mul_comm c] at h
  simpa [Int.mul_comm] using h

theorem sumFin_congr (n : Nat) (f g : Fin n → Int) (h : ∀ i, f i = g i) : sumFin n f = sumFin n g := by
  have : f = g := funext h
  rw [this]

theorem sumFin_comm (K n : Nat) (f : Fin K → Fin n → Int) :
    sumFin K (fun k => sumFin n (fun j => f k j)) = sumFin n (fun j => sumFin K (fun k => f k j)) := by
  induction n with
  | zero => simp [sumFin, sumFin_zero]
  | succ n ih =>
      have h := ih (fun k j => f k j.castSucc)
      simp only [sumFin] at h ⊢
      rw [sumFin_add, h]

def field {H : Type} (K : Nat) (base : H → Int) (atom : Fin K → H → Int) (z : Fin K → Int) (h : H) : Int :=
  base h + sumFin K (fun k => z k * atom k h)

theorem field_code_additive {H : Type} (K : Nat) (base : H → Int) (atom : Fin K → H → Int)
    (z w : Fin K → Int) (h : H) :
    field K base atom (fun k => z k + w k) h = field K base atom z h + field K base atom w h - base h := by
  unfold field
  have hs := sumFin_add K (fun k => z k * atom k h) (fun k => w k * atom k h)
  have he : (fun k => (z k + w k) * atom k h) = (fun k => z k * atom k h + w k * atom k h) := by
    funext k; exact Int.add_mul (z k) (w k) (atom k h)
  rw [he, hs]
  omega

theorem field_code_homogeneous {H : Type} (K : Nat) (base : H → Int) (atom : Fin K → H → Int)
    (c : Int) (z : Fin K → Int) (h : H) :
    field K base atom (fun k => c * z k) h - base h = c * (field K base atom z h - base h) := by
  unfold field
  have he : (fun k => c * z k * atom k h) = (fun k => c * (z k * atom k h)) := by
    funext k; exact Int.mul_assoc c (z k) (atom k h)
  have cancel : ∀ x y : Int, x + y - x = y := by intro x y; omega
  rw [he, sumFin_mul_left, cancel, cancel]

def linearAtom {n K : Nat} (W : Fin K → Fin n → Int) (k : Fin K) (h : Fin n → Int) : Int :=
  sumFin n (fun j => W k j * h j)

theorem linear_library_collapses (K n : Nat) (W : Fin K → Fin n → Int) (z : Fin K → Int) (h : Fin n → Int) :
    sumFin K (fun k => z k * linearAtom W k h) =
      sumFin n (fun j => sumFin K (fun k => z k * W k j) * h j) := by
  unfold linearAtom
  have step1 : (fun k => z k * sumFin n (fun j => W k j * h j)) =
      (fun k => sumFin n (fun j => z k * (W k j * h j))) := by
    funext k; rw [sumFin_mul_left]
  have step2 : (fun j => sumFin K (fun k => z k * W k j) * h j) =
      (fun j => sumFin K (fun k => z k * (W k j * h j))) := by
    funext j
    rw [← sumFin_mul_right]
    exact sumFin_congr K _ _ (fun k => Int.mul_assoc (z k) (W k j) (h j))
  rw [step1, step2, sumFin_comm]

theorem film_is_tied_rank_one_library (n : Nat) (w h c b : Fin n → Int) :
    sumFin n (fun j => w j * (h j * (1 + c j) + b j)) =
      sumFin n (fun j => w j * h j) + sumFin n (fun j => c j * (w j * h j)) + sumFin n (fun j => w j * b j) := by
  have pointwise : (fun j => w j * (h j * (1 + c j) + b j)) =
      (fun j => (w j * h j + c j * (w j * h j)) + w j * b j) := by
    funext j
    simp only [Int.mul_add, Int.mul_one]
    ac_rfl
  rw [pointwise, sumFin_add, sumFin_add]

def relu (x : Int) : Int := if x < 0 then 0 else x

theorem relu_not_affine : ¬ ∃ a b : Int, relu (-1) = a * (-1) + b ∧ relu 0 = a * 0 + b ∧ relu 1 = a * 1 + b := by
  intro ⟨a, b, h₁, h₀, h₂⟩
  simp [relu] at h₁ h₀ h₂
  omega

theorem relu_library_not_affine_in_input :
    ¬ ∃ a b : Int, ∀ x ∈ [(-1 : Int), 0, 1],
      field 1 (fun _ => 0) (fun _ x => relu x) (fun _ => 1) x = a * x + b := by
  intro ⟨a, b, h⟩
  have h₁ := h (-1) (by simp)
  have h₀ := h 0 (by simp)
  have h₂ := h 1 (by simp)
  simp [field, sumFin, relu] at h₁ h₀ h₂
  omega

def absI (x : Int) : Int := if x < 0 then -x else x

theorem absI_nonneg (x : Int) : 0 ≤ absI x := by
  unfold absI; split <;> omega

theorem mul_self_nonneg' (z : Int) : 0 ≤ z * z := by
  by_cases hz : 0 ≤ z
  · exact Int.mul_nonneg hz hz
  · have hn : 0 ≤ -z := by omega
    have := Int.mul_nonneg hn hn
    rwa [Int.neg_mul_neg] at this

theorem soft_threshold_prunes (a b lam : Int) (ha : 0 ≤ a) (hb : absI b ≤ lam) (z : Int) :
    a * (0 * 0) - 2 * (b * 0) + 2 * (lam * absI 0) ≤ a * (z * z) - 2 * (b * z) + 2 * (lam * absI z) := by
  have hq : 0 ≤ a * (z * z) := Int.mul_nonneg ha (mul_self_nonneg' z)
  have lin : b * z ≤ lam * absI z := by
    unfold absI at hb ⊢
    by_cases hz : z < 0
    · simp only [hz, if_true]
      have hl : -b ≤ lam := by split at hb <;> omega
      have h1 : 0 ≤ (lam + b) * (-z) := Int.mul_nonneg (by omega) (by omega)
      rw [Int.add_mul, Int.mul_neg, Int.mul_neg] at h1
      rw [Int.mul_neg]
      omega
    · simp only [hz, if_false]
      have hl : b ≤ lam := by split at hb <;> omega
      have h1 : 0 ≤ (lam - b) * z := Int.mul_nonneg (by omega) (by omega)
      rw [Int.sub_mul] at h1
      omega
  have h0 : absI 0 = 0 := rfl
  rw [h0]
  simp only [Int.mul_zero]
  omega

theorem rescale_output_invariant (z psi c : Int) : z * (c * psi) = (c * z) * psi := by
  rw [Int.mul_left_comm, Int.mul_assoc]

theorem gain_weighted_penalty_invariant (z psi c : Int) :
    (c * psi).natAbs * z.natAbs = psi.natAbs * (c * z).natAbs := by
  rw [Int.natAbs_mul, Int.natAbs_mul, Nat.mul_assoc, Nat.mul_left_comm]

theorem bare_code_penalty_loophole :
    (2 : Int) * 3 = 1 * 6 ∧ (1 : Int).natAbs < (2 : Int).natAbs := by decide

theorem collinear_atoms_not_identifiable (z1 z2 c t psi : Int) :
    (z1 + c * t) * psi + (z2 - t) * (c * psi) = z1 * psi + z2 * (c * psi) := by
  rw [Int.add_mul, Int.sub_mul, Int.mul_assoc c t psi, Int.mul_left_comm t c psi]
  omega

theorem gram_det_max_at_orthogonal (p q r : Int) : p * q - r * r ≤ p * q := by
  have := mul_self_nonneg' r
  omega

theorem trace_inverse_min_at_orthogonal (p q r : Int) (hp : 0 ≤ p) (hq : 0 ≤ q) :
    (p + q) * (p * q - r * r) ≤ (p + q) * (p * q) :=
  Int.mul_le_mul_of_nonneg_left (gram_det_max_at_orthogonal p q r) (by omega)

theorem descriptor_code_preserves_affine (F : Nat) (W d₁ d₂ : Fin F → Int) (α β : Int) :
    sumFin F (fun f => W f * (α * d₁ f + β * d₂ f)) =
      α * sumFin F (fun f => W f * d₁ f) + β * sumFin F (fun f => W f * d₂ f) := by
  have pointwise : (fun f => W f * (α * d₁ f + β * d₂ f)) =
      (fun f => α * (W f * d₁ f) + β * (W f * d₂ f)) := by
    funext f
    rw [Int.mul_add, Int.mul_left_comm (W f) α, Int.mul_left_comm (W f) β]
  rw [pointwise, sumFin_add, sumFin_mul_left, sumFin_mul_left]

theorem convex_mixture_bounded (e1 e2 w1 w2 N : Int) (h12 : e1 ≤ e2) (hw1 : 0 ≤ w1) (hw2 : 0 ≤ w2)
    (hN : w1 + w2 = N) : N * e1 ≤ w1 * e1 + w2 * e2 ∧ w1 * e1 + w2 * e2 ≤ N * e2 := by
  have d1 : 0 ≤ w2 * (e2 - e1) := Int.mul_nonneg hw2 (by omega)
  have d2 : 0 ≤ w1 * (e2 - e1) := Int.mul_nonneg hw1 (by omega)
  rw [Int.mul_sub] at d1 d2
  subst hN
  rw [Int.add_mul, Int.add_mul]
  constructor <;> omega

theorem convex_mixture_cannot_extrapolate (e1 e2 N : Int) (h12 : e1 < e2) (hN : 0 < N) :
    ¬ ∃ w1 w2 : Int, 0 ≤ w1 ∧ 0 ≤ w2 ∧ w1 + w2 = N ∧ w1 * e1 + w2 * e2 = N * (2 * e1 - e2) := by
  intro ⟨w1, w2, hw1, hw2, hsum, hout⟩
  have hb := (convex_mixture_bounded e1 e2 w1 w2 N (by omega) hw1 hw2 hsum).1
  have hlt : N * e1 < N * e2 := Int.mul_lt_mul_of_pos_left h12 hN
  rw [hout, Int.mul_sub, Int.mul_left_comm N 2 e1] at hb
  omega

end SALTheory
