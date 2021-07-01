### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 97539bb6-c0b0-11eb-2d83-316749d179f1
begin
	first_setup = false
	
	using Pkg
	Pkg.activate(".")
	if first_setup
		Pkg.add([
			"Pluto", "PlutoUI", "PlutoSliderServer",
			"Flux",
			"MLJ", "MLJFlux",
			"DifferentialEquations", "DiffEqFlux", 
			"MLDatasets", "MLBase",
			"Plots", "PlotlyJS",
			"ProgressMeter",
			"Statistics",
		])
		Pkg.update()
		Pkg.build()
		Pkg.precompile()
	else
		Pkg.instantiate()
	end

	using 
		Pluto, PlutoUI, PlutoSliderServer,
		Flux,
		MLJ, MLJFlux,
		DifferentialEquations, DiffEqFlux,
		MLDatasets, MLBase,
		Plots, PlotlyJS,
		ProgressMeter,
		Statistics
end

# ╔═╡ baab9408-fe06-436d-8158-6ec55e1b49cf
begin
	# MLJ.default_resource(CUDALibs())
	# MLJ.default_resource(CPUProcesses())
	# MLJ.default_resource(CPUThreads())
	MLJ.default_resource(CPU1())

	plotlyjs()
end

# ╔═╡ e593f5a3-9c57-42e7-9ba5-fcf4f0c4bb76
begin
	n_sam = 100
	train_x, train_y = MNIST.traindata()
	test_x,  test_y  = MNIST.testdata()

	train_x = coerce(train_x, GrayImage)[1:n_sam]
	train_y = coerce(train_y, Multiclass)[1:n_sam]
	test_x = coerce(test_x, GrayImage)[1:n_sam]
	test_y = coerce(test_y, Multiclass)[1:n_sam]

	@assert scitype(train_x) <: AbstractVector{<:GrayImage}
	@assert scitype(train_y) <: AbstractVector{<:Multiclass}
	@assert scitype(test_x) <: AbstractVector{<:GrayImage}
	@assert scitype(test_y) <: AbstractVector{<:Multiclass}
end

# ╔═╡ 3d5dac10-a6a9-4e13-b6c7-85c86d5f61ac
begin
	mutable struct ImgShortDense <: MLJFlux.Builder
		n_hidden_l::Int
		dropout::Float64
		σ
	end
	ImgShortDense(; n_hidden_l=1, dropout=1/8, σ=tanh) = ImgShortDense(n_hidden_l, dropout, σ)
	function MLJFlux.build(builder::ImgShortDense, rng, n, m, c)
		n = prod(n)
		
		ni_p = floor(Int, log2(n))
		no_p = ceil(Int, log2(m))
		n_middles = [2 ^ i for i in no_p:ni_p]
		
		imgsize_afterconvs = (n, 1)
		
		layers = [
		    # x -> float(cat(x..., dims=4)),
			Flux.flatten,
		    BatchNorm(prod(imgsize_afterconvs), builder.σ),
            Dropout(builder.dropout),
			Flux.Dense(prod(imgsize_afterconvs), n_middles[builder.n_hidden_l], builder.σ),
			BatchNorm(n_middles[builder.n_hidden_l], builder.σ),
            Dropout(builder.dropout),
		]
		for i in reverse(2:builder.n_hidden_l)
		    layers = vcat(layers, [
		        Flux.Dense(n_middles[i], n_middles[i-1], builder.σ),
			    BatchNorm(n_middles[i-1], builder.σ),
                Dropout(builder.dropout),
		    ])
	    end
	    layers = vcat(layers, [
	        Flux.Dense(n_middles[1], m, builder.σ),
		    BatchNorm(m, builder.σ),
            Dropout(builder.dropout),
	    ])

		return Flux.Chain(layers...) |> f64
	end
end

# ╔═╡ b48567da-38c0-4d01-b866-08413433d115
begin
	mutable struct ImgShortLite <: MLJFlux.Builder
		σ
	end
	ImgShortLite(; σ=tanh) = ImgShortLite(σ)
	function MLJFlux.build(builder::ImgShortLite, rng, n, m, c)
		n = prod(n)
		imgsize_afterconvs = (13, 13, 16)

		return Flux.Chain(
			# x -> float(cat(x..., dims=4)),
			Conv((3, 3), c => 16, builder.σ), # (26, 26, 16, N)
			MaxPool((2, 2)), # (13, 13, 16, N)
			Flux.flatten,	
			Flux.Dense(prod(imgsize_afterconvs), m, builder.σ),
		) |> f64
	end
end

# ╔═╡ 85bb6d5a-5117-4b24-beb1-c1b00f0e9462
begin
	mutable struct ImgShortConv <: MLJFlux.Builder
		dropout::Float64
		σ
	end
	ImgShortConv(; dropout=1/8, σ=tanh) = ImgShortConv(dropout, σ)
	function MLJFlux.build(builder::ImgShortConv, rng, n, m, c)
		return Flux.Chain(
			# x -> float(cat(x..., dims=4)),
			Conv((3, 3), c => 16, builder.σ), # (26, 26, 16, N)
			MaxPool((2, 2)), # (13, 13, 16, N)
			Conv((3, 3), 16 => 32, builder.σ), # (11, 11, 32, N)
			MaxPool((2, 2)), # (5, 5, 32, N)
			Conv((3, 3), 32 => 32, builder.σ), # (3, 3, 32, N)
			MaxPool((2, 2)), # (1, 1, 32, N)
			Flux.flatten,
			BatchNorm(32, builder.σ),
			Dropout(builder.dropout),
			Dense(32, 64, builder.σ),
			BatchNorm(64, builder.σ),
			Dropout(builder.dropout),
			Dense(64, 32, builder.σ),
			BatchNorm(32, builder.σ),
			Dropout(builder.dropout),
			NeuralODE(Chain(
				BatchNorm(32, builder.σ),
				Dropout(builder.dropout),
				Dense(32, 16, builder.σ),
				BatchNorm(16, builder.σ),
				Dropout(builder.dropout),
				Dense(16, 16, builder.σ),
				BatchNorm(16, builder.σ),
				Dropout(builder.dropout),
				Dense(16, 32, builder.σ),
				BatchNorm(32, builder.σ),
				Dropout(builder.dropout),
			) |> f64, 0.0:1.0, Tsit5()),
			x -> first(x.u),
		    BatchNorm(32, builder.σ),
            Dropout(builder.dropout),
			Flux.Dense(32, 16, builder.σ),
			BatchNorm(16, builder.σ),
            Dropout(builder.dropout),
			Flux.Dense(16, m, builder.σ),
		    BatchNorm(m, builder.σ),
            Dropout(builder.dropout),
		) |> f64
	end
end

# ╔═╡ 984cbe16-be12-4fdf-8345-65b601b2c4cd
begin
	itd = ImgShortDense()
	mitd = MLJFlux.build(itd, nothing, (28, 28), 10, 1)
	mitd(float(cat(train_x[1:10]..., dims=4)))
end

# ╔═╡ 4a988da3-1a6c-4bb0-a15c-779e2f2b5cee
begin
	itc = ImgShortConv()
	mitc = MLJFlux.build(itc, nothing, (28, 28), 10, 1)
	mitc(float(cat(train_x[1:10]..., dims=4)))
end

# ╔═╡ 03dd99ff-edf2-4c66-ac15-a4dff9b2c7c6
begin
	itl = ImgShortLite()
	mitl = MLJFlux.build(itc, nothing, (28, 28), 10, 1)
	mitl(float(cat(train_x[1:10]..., dims=4)))
end

# ╔═╡ 58ffa080-cf63-43d9-bc40-d9fedb085755
const ImgShort = ImgShortLite

# ╔═╡ d0e5c847-6bd5-484d-9e62-24deb3a8f0a5
begin
	losses = []
	training_losses = []

	add_loss(loss) = push!(losses, loss)
	add_training_loss(losses) = push!(training_losses, losses[end])
	
	f_mdl = IteratedModel(
		model=ImageClassifier(
			builder=ImgShort(),
			batch_size=2^5,
			acceleration=CPU1(),
		),
		controls=[
			Step(),
			TimeLimit(),
			NumberLimit(),
			NumberSinceBest(),
			Threshold(),
			GL(),
			PQ(),
			Patience(),

			# NotANumber(),
			# InvalidValue(),
			
			# Info(),
			# Warn(Bool),
			# Error(Bool),
			Callback(),
			WithNumberDo(),
			WithIterationsDo(),
			WithLossDo(),
			WithTrainingLossesDo(),
			
			WithLossDo(add_loss),
			WithTrainingLossesDo(add_training_loss),
			# Save(joinpath(@__DIR__, "mnist_machine.jlso")),
		],
		resampling=Holdout(),
		measure=cross_entropy,
		retrain=true,
	)
	# f_ranges = [
	# 	range(f_mdl, :(model.builder.n_hidden_l), lower=1, upper=6, scale=:linear),
	# 	range(f_mdl, :(model.builder.dropout), lower=0, upper=1, scale=:linear),
	# ]
	f_mach1 = machine(
		f_mdl,
		train_x,
		train_y,
	)
	# f_mach2 = machine(
	# 	TunedModel(
	# 		model=f_mdl,
	# 		tuning=Grid(
	# 			resolution=3,
	# 		),
	# 		resampling=Holdout(),
	# 		measure=cross_entropy,
	# 		range=f_ranges,
	# 		acceleration=CPU1(),
	# 		acceleration_resampling=CPU1(),
	# 	),
	# 	train_x,
	# 	train_y,
	# )
end

# ╔═╡ a1f4cd40-654b-4586-8bf0-3729b8415d3b
f_mach1

# ╔═╡ 08910ec3-1495-4e94-bdee-785c6148f1ee
fit!(f_mach1)

# ╔═╡ 016eb179-7add-4797-be11-a6011a7ec57a
report(f_mach1)

# ╔═╡ 0dd8e8a5-afe1-492f-a4f5-fb37ea15564e
f_mach1

# ╔═╡ 48b1b6c9-9beb-49fd-94d4-831712bccdb3
predicted_labels = MLJ.predict(f_mach1)

# ╔═╡ d54771fc-2ece-44b6-bfd7-138ebf60bb56
cross_entropy(predicted_labels, train_y) |> mean

# ╔═╡ 3d6d9bf8-c1f9-4089-bc72-301331ed749a
evaluate!(f_mach1, measure=cross_entropy)

# ╔═╡ 9c847976-1d10-4d8c-83dd-a9cc6600febc
# begin
# 	r = report(f_mach1).plotting
# 	fr = hcat(r.parameter_values, -1 .* r.measurements)
	
# 	Statistics.cor(fr)
# end

# ╔═╡ 9b60e4c1-c29a-49d6-846c-b2c7a942b9ca
# begin
# 	curve = report(f_mach1).plotting
# 	Plots.scatter(
# 		(x -> x[1]).(eachrow(curve.parameter_values)),
# 		(x -> x[2]).(eachrow(curve.parameter_values)),
# 		curve.measurements,
#      	xlab=curve.parameter_names[1],
# 		ylab=curve.parameter_names[2],
#      	xscale=curve.parameter_scales[1],
# 		yscale=curve.parameter_scales[2],
# 		zlab="CV estimate of error",
# 		# st=:surface,
# 	)
# end

# ╔═╡ a55dd90f-a2d1-48e5-97dd-d9afc91aa38a
size(training_losses), size(losses)

# ╔═╡ 4a9a00f0-6a97-436c-a483-77db50a123de
begin
	Plots.plot(
		losses,
		title="Cross Entropy",
		xlab = "epoch",
		label="out-of-sample",
	)
	Plots.plot!(
		training_losses,
		label="training",
	)
end

# ╔═╡ Cell order:
# ╠═97539bb6-c0b0-11eb-2d83-316749d179f1
# ╠═baab9408-fe06-436d-8158-6ec55e1b49cf
# ╠═e593f5a3-9c57-42e7-9ba5-fcf4f0c4bb76
# ╠═3d5dac10-a6a9-4e13-b6c7-85c86d5f61ac
# ╠═b48567da-38c0-4d01-b866-08413433d115
# ╠═85bb6d5a-5117-4b24-beb1-c1b00f0e9462
# ╠═984cbe16-be12-4fdf-8345-65b601b2c4cd
# ╠═4a988da3-1a6c-4bb0-a15c-779e2f2b5cee
# ╠═03dd99ff-edf2-4c66-ac15-a4dff9b2c7c6
# ╠═58ffa080-cf63-43d9-bc40-d9fedb085755
# ╠═d0e5c847-6bd5-484d-9e62-24deb3a8f0a5
# ╠═a1f4cd40-654b-4586-8bf0-3729b8415d3b
# ╠═08910ec3-1495-4e94-bdee-785c6148f1ee
# ╠═016eb179-7add-4797-be11-a6011a7ec57a
# ╠═0dd8e8a5-afe1-492f-a4f5-fb37ea15564e
# ╠═48b1b6c9-9beb-49fd-94d4-831712bccdb3
# ╠═d54771fc-2ece-44b6-bfd7-138ebf60bb56
# ╠═3d6d9bf8-c1f9-4089-bc72-301331ed749a
# ╠═9c847976-1d10-4d8c-83dd-a9cc6600febc
# ╠═9b60e4c1-c29a-49d6-846c-b2c7a942b9ca
# ╠═a55dd90f-a2d1-48e5-97dd-d9afc91aa38a
# ╠═4a9a00f0-6a97-436c-a483-77db50a123de
