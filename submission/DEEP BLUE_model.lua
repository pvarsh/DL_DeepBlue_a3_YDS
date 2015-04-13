
function run(sentence)
    return 5
end


function main()
    -- Prepare for input
    local sentence
    local N
    N = io.read()

    -- Load glove vector stuff
    

    -- Run the model 
    for i=1,N do
        sentence = io.read()
        rating = run(sentence)
        print(rating)
    end
end

main()